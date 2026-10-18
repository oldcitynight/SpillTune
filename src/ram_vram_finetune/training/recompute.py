# Adapted from AllTheAIs streaming/autograd.py; see SOURCE_ATTRIBUTION.md.
"""
streaming/autograd.py — 流式重算 backward (Spec 3.2)

Forward 和 backward 重算使用完全相同的函数图（同一纯函数入口）。
Projection-level LoRA 在两条路径中的输入、缩放、dtype 和运算顺序完全一致。
"""
from typing import Callable, Optional, cast
from contextlib import contextmanager

import torch

from ..finetune.lora import LoRAProjection


class StreamingBlockFunction(torch.autograd.Function):
    """通用流式 block 的 forward + backward。

    Forward: 调用纯函数入口 func_block(...)
    Backward: 从 pinned 重新加载权重 → 调用同一 func_block → torch.autograd.grad()

    这保证 forward 和 backward 重算使用完全相同的代码路径。
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor],

        # 纯函数入口
        func_block: Callable,
        func_kwargs: dict,

        # 权重重载器
        weight_reloader: Callable[[], dict[str, torch.Tensor]],

        # LoRA adapters
        lora_dict: dict[str, LoRAProjection],

        # block index
        block_idx: int,

        # Explicit micro-batch accumulation control.  False means this
        # backward replaces the adapter buffers; True adds to them.
        accumulate_adapter_grads: bool = False,
    ) -> torch.Tensor:
        _validate_lora_bindings(func_kwargs, lora_dict)
        ctx.func_block = func_block
        ctx.func_kwargs = func_kwargs
        ctx.weight_reloader = weight_reloader
        ctx.lora_dict = lora_dict
        ctx.block_idx = block_idx
        ctx.accumulate_adapter_grads = accumulate_adapter_grads
        ctx.position_embeddings = position_embeddings
        ctx.attention_mask = attention_mask

        # 保存输入供 backward 使用
        ctx.save_for_backward(hidden_states.detach().clone())

        # Forward: 调用纯函数入口
        with _borrow_weights(weight_reloader, "forward") as weights:
            output = _call_functional_block(
                func_block,
                hidden_states,
                position_embeddings,
                attention_mask,
                weights,
                func_kwargs,
            )
            return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        hidden_states = ctx.saved_tensors[0].detach().requires_grad_(True)

        # 收集需要梯度的 LoRA 参数
        lora_tensors = []
        for proj in sorted(ctx.lora_dict.values(), key=lambda p: p.name):
            lora_tensors.append(proj.A)
            lora_tensors.append(proj.B)
        all_inputs = [hidden_states] + lora_tensors

        with _borrow_weights(ctx.weight_reloader, "backward") as weights:
            # 重算: 调用同一纯函数入口
            with torch.enable_grad():
                output = _call_functional_block(
                    ctx.func_block,
                    hidden_states,
                    ctx.position_embeddings,
                    ctx.attention_mask,
                    weights,
                    ctx.func_kwargs,
                )

            # 计算所有梯度
            # Torch's runtime returns ``None`` for disconnected inputs when
            # ``allow_unused=True`` even though its current type stub declares only
            # tensors.  Preserve the real contract so the integrity checks below
            # remain reachable and type-checked.
            grads = cast(
                tuple[Optional[torch.Tensor], ...],
                torch.autograd.grad(output, all_inputs, grad_output, allow_unused=True),
            )

            # 提取并累积梯度
            grad_input = grads[0]
            if grad_input is None:
                raise RuntimeError(f"block {ctx.block_idx} lost its input gradient")
            grad_idx = 1
            for proj in sorted(ctx.lora_dict.values(), key=lambda p: p.name):
                grad_a, grad_b = grads[grad_idx], grads[grad_idx + 1]
                if grad_a is None or grad_b is None:
                    missing = [
                        name
                        for name, gradient in (("A", grad_a), ("B", grad_b))
                        if gradient is None
                    ]
                    raise RuntimeError(
                        f"block {ctx.block_idx} adapter {proj.name} is disconnected; "
                        f"missing gradients for {missing}"
                    )
                if not ctx.accumulate_adapter_grads:
                    proj.zero_grad()
                proj.grad_A.add_(grad_a.detach().to(proj.grad_A))
                proj.grad_B.add_(grad_b.detach().to(proj.grad_B))
                grad_idx += 2


        # 返回 None for non-tensor args
        # forward has 9 arguments after ctx.
        return (grad_input,) + (None,) * 8


@contextmanager
def _borrow_weights(reloader, phase):
    borrow = getattr(reloader, 'borrow', None)
    if borrow is None:
        yield reloader()
    else:
        with borrow(phase) as weights:
            yield weights


def _call_functional_block(
    func_block: Callable,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],
    weights: dict[str, torch.Tensor],
    func_kwargs: dict,
) -> torch.Tensor:
    """Call the single block graph with a strict, deterministic interface."""
    reserved = {"hidden_states", "position_embeddings", "attention_mask"}
    duplicate_reserved = reserved.intersection(func_kwargs)
    if duplicate_reserved:
        raise TypeError(
            "func_kwargs duplicates StreamingBlockFunction arguments: "
            f"{sorted(duplicate_reserved)}"
        )
    overlap = set(weights).intersection(func_kwargs)
    if overlap:
        raise TypeError(f"weight/functional kwargs overlap: {sorted(overlap)}")
    for name, weight in weights.items():
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"reloaded weight {name!r} is not a Tensor")
        if weight.requires_grad:
            raise ValueError(f"base weight {name!r} must remain frozen")

    return func_block(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        **weights,
        **func_kwargs,
    )


def _validate_lora_bindings(
    func_kwargs: dict,
    lora_dict: dict[str, LoRAProjection],
) -> None:
    """Prevent the functional graph and optimizer from binding different A/B."""
    if not isinstance(lora_dict, dict):
        raise TypeError("lora_dict must be a dict[str, LoRAProjection]")
    expected = list(lora_dict.values())
    if any(not isinstance(projection, LoRAProjection) for projection in expected):
        raise TypeError("every lora_dict value must be a LoRAProjection")
    bound = [value for value in func_kwargs.values() if isinstance(value, LoRAProjection)]
    expected_ids = [id(value) for value in expected]
    bound_ids = [id(value) for value in bound]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("lora_dict contains the same projection more than once")
    if len(bound_ids) != len(set(bound_ids)):
        raise ValueError("functional kwargs bind the same LoRA projection more than once")
    if set(expected_ids) != set(bound_ids):
        raise ValueError(
            "functional LoRA bindings must be the exact projections exposed "
            "to the optimizer/gradient buffers"
        )


def streaming_block(
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],
    func_block: Callable,
    func_kwargs: dict,
    weight_reloader: Callable[[], dict[str, torch.Tensor]],
    lora_dict: dict[str, LoRAProjection],
    block_idx: int,
    *,
    accumulate_adapter_grads: bool = False,
) -> torch.Tensor:
    """Typed wrapper around :meth:`StreamingBlockFunction.apply`."""
    return StreamingBlockFunction.apply(
        hidden_states,
        position_embeddings,
        attention_mask,
        func_block,
        func_kwargs,
        weight_reloader,
        lora_dict,
        block_idx,
        accumulate_adapter_grads,
    )


# The two architecture labels intentionally share one implementation and one
# signature; block-specific math is supplied by the functional callable.
StreamingFullAttentionFunction = StreamingBlockFunction
StreamingLinearAttentionFunction = StreamingBlockFunction
