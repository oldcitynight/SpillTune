# Adapted from AllTheAIs streaming/functional_blocks.py; see SOURCE_ATTRIBUTION.md.
"""
streaming/functional_blocks.py — 统一 block 计算入口 (Spec 3.2)

每个 block type 有唯一纯函数入口，forward 和 backward 重算均调用同一入口。
Projection-level LoRA 注入在每条 F.linear 调用处。
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F

from ...finetune.lora import LoRAProjection


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3.5 RMSNorm, whose stored weight is an offset from one."""
    original_dtype = x.dtype
    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
    return (normalized * (1.0 + weight.float())).to(original_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3.5 partial rotary embeddings."""
    del position_ids  # cos/sin are already indexed by the caller
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    q_embed = q_rot * cos + rotate_half(q_rot) * sin
    k_embed = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)


def _linear_with_lora(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    lora: Optional[LoRAProjection],
) -> torch.Tensor:
    """Apply LoRA at the input of one concrete linear projection.

    The base projection follows the model dtype.  A/B remain FP32 master
    tensors and the differentiable casts below make autograd accumulate FP32
    gradients on those masters.
    """
    out = F.linear(x, weight, bias)
    if lora is not None:
        if tuple(weight.shape) != (lora.out_features, lora.in_features):
            raise ValueError(
                f"LoRA/base shape mismatch for {lora.name}: base={tuple(weight.shape)}, "
                f"expected={(lora.out_features, lora.in_features)}"
            )
        lora_input = x.to(dtype=lora.A.dtype)
        update = F.linear(F.linear(lora_input, lora.A), lora.B)
        # Preserve the specified operation order: scale the FP32 update before
        # the model-dtype cast, then accumulate with the frozen base result.
        out = (out.float() + update * lora.scale).to(dtype=out.dtype)
    return out


def prepare_linear_attention_core(module: torch.nn.Module) -> tuple[torch.nn.Module, torch.Tensor]:
    """Return an independent HF linear-attention core without ``out_proj``.

    Qwen3.5's GatedDeltaNet does not expose the tensor entering ``out_proj``.
    Replacing that one projection with ``Identity`` lets the unique functional
    block apply both the frozen base weight and LoRA to the *same real
    projection input*.  The original module is not mutated.
    """
    source_out_proj = getattr(module, "out_proj", None)
    if not isinstance(source_out_proj, torch.nn.Linear):
        raise TypeError("linear-attention module must expose an nn.Linear out_proj")
    core = copy.deepcopy(module)
    core_out_proj = getattr(core, "out_proj", None)
    if not isinstance(core_out_proj, torch.nn.Linear):
        raise TypeError("copied linear-attention core lost its nn.Linear out_proj")
    out_weight = core_out_proj.weight.detach().clone()
    setattr(core, "out_proj", torch.nn.Identity())
    core.requires_grad_(False)
    return core, out_weight


# ── Full Attention Block ───────────────────────────────────

def functional_full_attention_block_v2(
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],

    # Base weights (frozen, from pinned store)
    q_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,

    # LoRA adapters (one per target projection, optional)
    lora_q: Optional[LoRAProjection] = None,
    lora_v: Optional[LoRAProjection] = None,
    lora_o: Optional[LoRAProjection] = None,

    # Config
    num_heads: int = 16,
    num_kv_heads: int = 4,
    head_dim: int = 256,
    eps: float = 1e-6,
) -> torch.Tensor:
    """统一 full_attention block 纯函数入口。

    由 forward 和 backward 重算共用。LoRA 注入在每个 projection 处。
    """
    cos, sin = position_embeddings
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)
    residual = hidden_states

    # RMS Norm
    normed = rms_norm(hidden_states, input_layernorm_weight, eps)

    # Q projection + gate split (exact Qwen3.5 pattern).  The adapter belongs
    # here, before reshape/chunk, because q_proj is the real linear module.
    q_all = _linear_with_lora(normed, q_proj_weight, None, lora_q)
    q_all = q_all.view(*input_shape, -1, head_dim * 2)
    query_states, gate = torch.chunk(q_all, 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    # Q norm + transpose
    query_states = query_states.view(hidden_shape)
    query_states = rms_norm(query_states.float(), q_norm_weight.float()).to(query_states.dtype)
    query_states = query_states.transpose(1, 2)  # [B, heads, S, head_dim]

    # K projection + norm
    key_states = F.linear(normed, k_proj_weight)
    key_states = key_states.view(hidden_shape)
    key_states = rms_norm(key_states.float(), k_norm_weight.float()).to(key_states.dtype)
    key_states = key_states.transpose(1, 2)

    # V projection (+ LoRA)
    if lora_v is not None:
        value_states = _linear_with_lora(normed, v_proj_weight, None, lora_v)
    else:
        value_states = F.linear(normed, v_proj_weight)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    # RoPE
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

    # GQA repeat
    if num_kv_heads < num_heads:
        key_states = key_states.repeat_interleave(num_heads // num_kv_heads, dim=1)
        value_states = value_states.repeat_interleave(num_heads // num_kv_heads, dim=1)

    # Attention
    scale = head_dim ** -0.5
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # Reshape + gate
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = attn_output * torch.sigmoid(gate)

    # O projection (+ LoRA)
    if lora_o is not None:
        attn_output = _linear_with_lora(attn_output, o_proj_weight, None, lora_o)
    else:
        attn_output = F.linear(attn_output, o_proj_weight)

    hidden_states = residual + attn_output

    # MLP
    residual = hidden_states
    normed = rms_norm(hidden_states, post_attention_layernorm_weight, eps)
    gate_mlp = F.linear(normed, gate_proj_weight)
    up = F.linear(normed, up_proj_weight)
    mlp_out = F.linear(F.silu(gate_mlp) * up, down_proj_weight)
    hidden_states = residual + mlp_out

    return hidden_states


# ── Linear Attention Block ──────────────────────────────────

def functional_linear_attention_block_v2(
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],

    # HF token-mixer core with out_proj replaced by Identity
    hf_linear_attn_core: torch.nn.Module,
    out_proj_weight: torch.Tensor,

    # MLP weights
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,

    # LoRA on out_proj
    lora_out: Optional[LoRAProjection] = None,

    eps: float = 1e-6,

    # Optional streamed GatedDeltaNet-core weights.  A production reloader
    # supplies the complete set; tiny/resident tests may use a preloaded core.
    in_proj_qkv_weight: Optional[torch.Tensor] = None,
    in_proj_a_weight: Optional[torch.Tensor] = None,
    in_proj_b_weight: Optional[torch.Tensor] = None,
    in_proj_z_weight: Optional[torch.Tensor] = None,
    conv1d_weight: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """统一 linear_attention block 纯函数入口。

    使用 HF fused CUDA kernel 计算 GatedDeltaNet core。基础 out_proj 与
    LoRA 都在本函数中作用于同一个 core 输出，因此不是 block-output
    adapter，也不会错误地使用 block 输入。
    """
    _load_linear_attention_core_weights(
        hf_linear_attn_core,
        {
            "in_proj_qkv.weight": in_proj_qkv_weight,
            "in_proj_a.weight": in_proj_a_weight,
            "in_proj_b.weight": in_proj_b_weight,
            "in_proj_z.weight": in_proj_z_weight,
            "conv1d.weight": conv1d_weight,
            "dt_bias": dt_bias,
            "A_log": A_log,
            "norm.weight": norm_weight,
        },
    )
    residual = hidden_states
    normed = rms_norm(hidden_states, input_layernorm_weight, eps)

    # HF kernel core forward.  Qwen3.5 GatedDeltaNet returns a Tensor; tolerate
    # tuple-returning compatible kernels without accidentally indexing batch 0.
    core_out = hf_linear_attn_core(normed, attention_mask=attention_mask)
    if isinstance(core_out, tuple):
        core_out = core_out[0]

    # The one real out_proj operation, with its projection-level adapter.
    attn_out = _linear_with_lora(core_out, out_proj_weight, None, lora_out)

    hidden_states = residual + attn_out

    # MLP
    residual = hidden_states
    normed = rms_norm(hidden_states, post_attention_layernorm_weight, eps)
    gate_mlp = F.linear(normed, gate_proj_weight)
    up = F.linear(normed, up_proj_weight)
    mlp_out = F.linear(F.silu(gate_mlp) * up, down_proj_weight)
    hidden_states = residual + mlp_out

    return hidden_states


def _load_linear_attention_core_weights(
    core: torch.nn.Module,
    streamed: dict[str, Optional[torch.Tensor]],
) -> None:
    """Load one complete streamed GatedDeltaNet state into the shared core."""
    supplied = {name: tensor for name, tensor in streamed.items() if tensor is not None}
    if not supplied:
        return
    if len(supplied) != len(streamed):
        missing = sorted(set(streamed) - set(supplied))
        raise ValueError(f"incomplete linear-attention core weight set: {missing}")

    parameters = dict(core.named_parameters())
    expected = set(streamed)
    missing_parameters = expected - set(parameters)
    if missing_parameters:
        raise KeyError(
            "linear-attention core does not expose streamed parameters: "
            f"{sorted(missing_parameters)}"
        )
    with torch.no_grad():
        for name, tensor in supplied.items():
            parameter = parameters[name]
            if parameter.requires_grad:
                raise ValueError(f"linear-attention base parameter must be frozen: {name}")
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"linear-attention core shape mismatch for {name}: "
                    f"{tuple(parameter.shape)} != {tuple(tensor.shape)}"
                )
            if parameter.dtype != tensor.dtype or parameter.device != tensor.device:
                raise TypeError(
                    f"linear-attention core dtype/device mismatch for {name}: "
                    f"{parameter.dtype}/{parameter.device} != {tensor.dtype}/{tensor.device}"
                )
            parameter.copy_(tensor)
