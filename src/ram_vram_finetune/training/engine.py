"""Architecture-independent causal LM engine over an adapter and a slot session."""
import hashlib
import json
import os
import random
from ..platform_support import rss_high_water_bytes, platform_capabilities
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import TrainConfig
from ..finetune.optimizer import LoRAOptimizer
from ..memory import SlotSession
from ..buffers import contract_bindings
from .recompute import streaming_block


def flatten(adapter):
    result = {}
    seen = set()
    for block in adapter.blocks:
        for name, proj in block.adapters.items():
            key = f"{block.index}:{name}"
            if key in result or id(proj) in seen:
                raise ValueError("unsupported_capability: duplicate/shared trainable parameter")
            result[key] = proj
            seen.add(id(proj))
    if not result:
        raise ValueError("training plan has no trainable projections")
    return result


def forward(adapter, session, hidden, positions, masks, *, accumulate=False, resident=False):
    for block in adapter.blocks:
        if resident:
            hidden = block.functional(hidden_states=hidden, position_embeddings=positions,
                                      attention_mask=masks[block.kind], **block.host_weights, **block.kwargs)
        else:
            hidden = streaming_block(hidden, positions, masks[block.kind], block.functional,
                                     block.kwargs, session.reloader(block), block.adapters,
                                     block.index, accumulate_adapter_grads=accumulate)
    return hidden


def microbatch(adapter, session, input_ids, labels, *, normalization, token_chunk, accumulate=False):
    """Exact shifted summed CE / window token count; decoder backward runs once."""
    device = adapter.device
    valid = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    positions, masks = adapter.positions_and_masks(valid)
    embedded = F.embedding(input_ids.cpu(), adapter.embedding).to(device).detach().requires_grad_(True)
    hidden = forward(adapter, session, embedded, positions, masks, accumulate=accumulate)
    # Preserve native full-sequence terminal-normalization VJP shape. Doing
    # norm separately on prefixes/chunks changes BF16 reduction rounding and
    # can amplify through a deep decoder, despite almost identical CE values.
    terminal_input = hidden.detach().requires_grad_(True)
    normalized = adapter.terminal_norm(terminal_input)
    normalized_gradient = torch.zeros_like(normalized)
    loss_sum = torch.zeros((), device=device)
    labels = labels.to(device)
    for start in range(0, input_ids.shape[1] - 1, token_chunk):
        end = min(start + token_chunk, input_ids.shape[1] - 1)
        target = labels[:, start + 1:end + 1]
        if not (target != -100).any():
            continue
        h = normalized[:, start:end].detach().requires_grad_(True)
        logits = F.linear(h, adapter.head)
        chunk = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1),
                                ignore_index=-100, reduction="sum")
        (grad,) = torch.autograd.grad(chunk / normalization, h)
        normalized_gradient[:, start:end].copy_(grad.detach())
        loss_sum += chunk.detach()
    (gradient,) = torch.autograd.grad(normalized, terminal_input, normalized_gradient)
    if not torch.isfinite(loss_sum) or not torch.isfinite(gradient).all():
        raise FloatingPointError("non-finite causal loss/gradient")
    hidden.backward(gradient)
    return float(loss_sum / normalization), embedded.grad.detach().clone()


def load_rows(config):
    raw = [json.loads(line) for line in Path(config.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw:
        raise ValueError("dataset is empty")
    if all(isinstance(row, dict) and set(row) == {"sample_id", "messages"} for row in raw):
        from ..data.chat import load_jsonl_dataset, load_real_tokenizer
        tokenizer = load_real_tokenizer(Path(config.model))
        return [(s.input_ids.unsqueeze(0), s.labels.unsqueeze(0))
                for s in load_jsonl_dataset(Path(config.dataset), tokenizer,
                                           max_sequence_length=config.max_sequence_length)]
    rows = []
    for row in raw:
        if set(row) != {"input_ids", "labels"}:
            raise ValueError("rows must have exactly input_ids/labels or sample_id/messages")
        ids, labels = row["input_ids"], row["labels"]
        if not isinstance(ids, list) or not isinstance(labels, list) or any(type(v) is not int for v in ids + labels):
            raise ValueError("token IDs and labels must be integer lists")
        if len(ids) != len(labels) or not 2 <= len(ids) <= config.max_sequence_length:
            raise ValueError("invalid token/label lengths")
        if any(v < 0 for v in ids) or any(v < -100 or -100 < v < 0 for v in labels):
            raise ValueError("invalid negative token/label")
        if not any(v != -100 for v in labels[1:]):
            raise ValueError("zero shifted supervised tokens")
        rows.append((torch.tensor([ids]), torch.tensor([labels])))
    return rows


def run(config: TrainConfig, *, resume=None, cancel_after_step=None, progress_callback=None):
    from ..adapters.registry import resolve
    from ..checkpoint_io.state import save_checkpoint, load_checkpoint, export_adapter
    # Explicit registry dispatch; unknown adapters are rejected by configuration.
    factory = resolve(config.adapter)
    random.seed(config.seed); torch.manual_seed(config.seed)
    rows = load_rows(config)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    adapter = session = None
    stop = [False]
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        if __import__("threading").current_thread() is __import__("threading").main_thread():
            old_handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    try:
        if config.device == 'cuda:0' and torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        quant_kwargs={'quantization':config.quantization,'quantized_compute':config.quantized_compute} if config.quantization!='none' else {}
        adapter = factory(config.model, rank=config.rank, alpha=config.alpha, device=config.device,
                                host_budget_bytes=config.host_budget_bytes,
                                device_budget_bytes=config.device_budget_bytes,**quant_kwargs)
        if progress_callback is not None:
            progress_callback("adapter_ready", {"adapter": adapter})
        for ids, labels in rows:
            if int(ids.max()) >= adapter.embedding.shape[0] or int(labels.max()) >= adapter.head.shape[0]:
                raise ValueError("token/label outside vocabulary")
        if config.transfer_mode == 'prefetch':
            from ..prefetch import PrefetchSession
            session = PrefetchSession(adapter.blocks, config.device, host_budget_bytes=config.host_budget_bytes,
                 device_budget_bytes=config.device_budget_bytes,pool_budget_bytes=config.prefetch_pool_budget_bytes,
                 max_slots=config.prefetch_max_slots,reserve_bytes=config.prefetch_reserve_bytes)
        else:
            session = SlotSession(adapter.blocks, config.device, host_budget_bytes=config.host_budget_bytes,
                                  device_budget_bytes=config.device_budget_bytes, transfer_mode=config.transfer_mode)
        init_peak_allocated = torch.cuda.max_memory_allocated(adapter.device) if adapter.device.type=='cuda' else 0
        init_peak_reserved = torch.cuda.max_memory_reserved(adapter.device) if adapter.device.type=='cuda' else 0
        projections = flatten(adapter)
        if progress_callback is not None:
            progress_callback("session_ready", {"adapter": adapter, "session": session})
        optimizer = LoRAOptimizer(lr=config.learning_rate, weight_decay=config.weight_decay, validation_mode=config.optimizer_validation)
        binding = {"model": adapter.identity, "dataset": hashlib.sha256(Path(config.dataset).read_bytes()).hexdigest(),
                   "tokenized_rows": hashlib.sha256(json.dumps([(ids.tolist(), labels.tolist()) for ids, labels in rows]).encode()).hexdigest(),
                   "config": config.semantics()}
        buffer_bindings = contract_bindings(adapter.blocks)
        if buffer_bindings:
            binding['typed_buffers'] = buffer_bindings
        cursor = 0
        if resume:
            cursor = load_checkpoint(resume, projections, optimizer, binding)
        if optimizer.step_count >= config.steps:
            raise ValueError("resume checkpoint already reached requested total steps")
        metrics = []
        for step in range(optimizer.step_count + 1, config.steps + 1):
            if stop[0]:
                break
            items = [rows[(cursor + i) % len(rows)] for i in range(config.accumulation)]
            normalization = sum(int((labels[:, 1:] != -100).sum()) for _, labels in items)
            optimizer.zero_grad(projections)
            before_reloads = session.reloads
            if adapter.device.type == "cuda":
                torch.cuda.synchronize(adapter.device)
                torch.cuda.reset_peak_memory_stats(adapter.device)
            started = time.perf_counter()
            loss = 0.0
            with session.run():
                for i, (ids, labels) in enumerate(items):
                    micro_loss, _ = microbatch(adapter, session, ids, labels, normalization=normalization,
                                               token_chunk=config.token_chunk, accumulate=i > 0)
                    loss += micro_loss
                optimizer.step(projections)
                if adapter.device.type == "cuda" and not getattr(session,'asynchronous',False):
                    torch.cuda.synchronize(adapter.device)
            cursor += config.accumulation
            metrics.append({"step": step, "loss": loss, "seconds": time.perf_counter() - started,
                            "reloads": session.reloads - before_reloads,
                            "host_weight_bytes": session.host_bytes, "slot_bytes": session.capacity,
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(adapter.device) if adapter.device.type == "cuda" else 0,
                            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(adapter.device) if adapter.device.type == "cuda" else 0,
                            "rss_high_water_bytes": rss_high_water_bytes()})
            if progress_callback is not None:
                progress_callback("step", {"adapter": adapter, "session": session, "metrics": metrics[-1]})
            if step % config.checkpoint_every == 0 or step == config.steps or stop[0] or step == cancel_after_step:
                save_checkpoint(output / f"checkpoint-{step}", projections, optimizer, binding, cursor)
            if stop[0] or step == cancel_after_step:
                stop[0] = True
                break
        status = "cancelled" if stop[0] else "completed"
        if progress_callback is not None:
            progress_callback("training_complete", {"adapter": adapter, "session": session})
        # A signal can arrive while a completion observer verifies large base
        # tensors. Recheck the safe-point flag before publishing adapters.
        if stop[0]:
            status = "cancelled"
        if status == "completed":
            export_adapter(output / "adapter", adapter, config)
        # Export is atomic and may complete after a signal. Keep its complete
        # artifacts, but never report completed after cancellation arrived.
        if stop[0]:
            status = 'cancelled'
        report = {"status": status, "optimizer_steps": optimizer.step_count, "cursor": cursor,
                  "model_identity": adapter.identity, "metrics": metrics,
                  "reloads": session.reloads, "logical_weight_copy_bytes": session.transferred_bytes,
                  "transfer_mode": "synchronous_single_slot" if config.transfer_mode == "synchronous" else ("budgeted_prefetch" if config.transfer_mode == "prefetch" else "batched_copy_single_slot"),
                  "optimizer_validation": config.optimizer_validation, "device": config.device,
                  "cuda_init_peak_allocated_bytes": init_peak_allocated,
                  "cuda_init_peak_reserved_bytes": init_peak_reserved,
                  "pinned_arena_capacity_bytes": session.pinned_capacity_bytes,
                  "quantization":getattr(adapter,'storage_metadata',None),
                  "upload_slot_bytes":session.upload_capacity,"actual_h2d_bytes":session.actual_h2d_bytes,
                  "gpu_validation": "EXECUTED" if adapter.device.type == "cuda" else "NOT_RUN", "data_order": "cyclic_input_order_no_shuffle",
                  "transfer_trace": session.trace,
                  "scheduler":session.runtime_metadata() if hasattr(session,"runtime_metadata") else None,
                  "adapter_runtime": adapter.runtime_metadata(),
                  "pid": os.getpid(), "platform_capabilities": platform_capabilities()}
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        return report
    finally:
        if session is not None:
            session.close()
        if adapter is not None:
            adapter.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
