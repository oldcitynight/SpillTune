import copy
import dataclasses
import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ram_vram_finetune.adapters.qwen35.adapter import Qwen35Adapter, inspect_model
from ram_vram_finetune.checkpoint_io.state import load_checkpoint, save_checkpoint, FIELDS
from ram_vram_finetune.config import TrainConfig
from ram_vram_finetune.finetune.optimizer import LoRAOptimizer
from ram_vram_finetune.fixture import make_tiny
from ram_vram_finetune.memory import SlotSession
from ram_vram_finetune.training.engine import flatten, microbatch, run


@pytest.fixture
def tiny(tmp_path):
    torch.set_num_threads(1)
    return make_tiny(tmp_path / "model")


def session(adapter):
    return SlotSession(adapter.blocks, "cpu", host_budget_bytes=1024**3, device_budget_bytes=1024**3)


def native_reference(adapter, model_path):
    """Independent HF decoder graph with projection-level trainables injected."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    native = Qwen3_5TextModel(copy.deepcopy(adapter.config)).float().eval().requires_grad_(False)
    state = load_file(str(model_path / "model.safetensors"))
    native.load_state_dict({k.removeprefix("model.language_model."): v for k, v in state.items()
                            if k.startswith("model.language_model.")}, strict=True)

    class Projection(torch.nn.Module):
        def __init__(self, original, projection):
            super().__init__()
            self.original = original
            self.projection = projection

        def forward(self, x):
            return self.projection.forward(x, self.original.weight)

    for block in adapter.blocks:
        for target, projection in block.adapters.items():
            parent_path, leaf = target.rsplit(".", 1)
            parent = native.layers[block.index].get_submodule(parent_path)
            setattr(parent, leaf, Projection(getattr(parent, leaf), projection))
    return native


def native_loss(native, adapter, ids, labels):
    positions, masks = adapter.positions_and_masks(torch.ones_like(ids, dtype=torch.bool))
    embedded = native.embed_tokens(ids).detach().requires_grad_(True)
    hidden = embedded
    for block, layer in zip(adapter.blocks, native.layers):
        hidden = layer(hidden, position_embeddings=positions, attention_mask=masks[block.kind])
    logits = F.linear(native.norm(hidden), adapter.head)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                           reduction="sum", ignore_index=-100)
    return loss, embedded


def compare(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("nonzero_b", [False, True])
def test_native_hf_loss_input_and_every_adapter_gradient(tiny, nonzero_b, record_property):
    adapter = Qwen35Adapter(tiny)
    if nonzero_b:
        with torch.no_grad():
            for p in flatten(adapter).values():
                p.B.normal_(0, 0.03)
    native = native_reference(adapter, tiny)
    projections = flatten(adapter)
    tensors = [t for p in projections.values() for t in (p.A, p.B)]
    ids, labels = torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[-100, -100, 3, 4, 5]])
    reference, embedded = native_loss(native, adapter, ids, labels)
    reference = reference / 3
    expected = torch.autograd.grad(reference, [embedded, *tensors])
    slot = session(adapter)
    with slot.run():
        loss, input_grad = microbatch(adapter, slot, ids, labels, normalization=3, token_chunk=2)
    assert loss == pytest.approx(float(reference.detach()), rel=2e-6, abs=2e-6)
    compare(input_grad, expected[0])
    for p, ga, gb in zip(projections.values(), expected[1::2], expected[2::2]):
        compare(p.grad_A, ga); compare(p.grad_B, gb)
    actual = [input_grad, *[t for p in projections.values() for t in (p.grad_A, p.grad_B)]]
    record_property("max_abs_input_adapter_grad_error", max(float((a - b).abs().max()) for a, b in zip(actual, expected)))
    record_property("loss_abs_error", abs(loss - float(reference.detach())))
    assert slot.reloads == 2 * len(adapter.blocks)
    if not nonzero_b:
        assert all(torch.count_nonzero(p.grad_A) == 0 for p in projections.values())
    slot.close(); adapter.close()


def test_accumulation_and_two_adamw_updates_match_native(tiny):
    adapter = Qwen35Adapter(tiny)
    # Native reference owns independent A/B buffers and optimizer.
    reference_adapter = Qwen35Adapter(tiny)
    for dst, src in zip(flatten(reference_adapter).values(), flatten(adapter).values()):
        with torch.no_grad():
            dst.A.copy_(src.A); dst.B.copy_(src.B)
    native = native_reference(reference_adapter, tiny)
    projections = flatten(adapter)
    reference_projections = flatten(reference_adapter)
    reference_tensors = [t for p in reference_projections.values() for t in (p.A, p.B)]
    expected_optimizer = torch.optim.AdamW(reference_tensors, lr=1e-3, weight_decay=0.1, foreach=False)
    optimizer = LoRAOptimizer(lr=1e-3, weight_decay=0.1)
    slot = session(adapter)
    original = {f"{b.index}/{k}": v.clone() for b in adapter.blocks for k, v in b.host_weights.items()}
    items = [(torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[-100, -100, 3, 4, 5]])),
             (torch.tensor([[3, 6, 9]]), torch.tensor([[-100, -100, 9]]))]
    for _ in range(2):
        expected_optimizer.zero_grad(set_to_none=True)
        loss = sum(native_loss(native, reference_adapter, *item)[0] for item in items) / 4
        loss.backward()
        optimizer.zero_grad(projections)
        with slot.run():
            losses = [microbatch(adapter, slot, *item, normalization=4, token_chunk=1,
                                accumulate=i > 0)[0] for i, item in enumerate(items)]
            assert sum(losses) == pytest.approx(float(loss.detach()), rel=3e-6)
            for p, r in zip(projections.values(), reference_projections.values()):
                compare(p.grad_A, r.A.grad); compare(p.grad_B, r.B.grad)
            optimizer.step(projections)
        expected_optimizer.step()
        for p, r in zip(projections.values(), reference_projections.values()):
            compare(p.A, r.A); compare(p.B, r.B)
            compare(p.m_A, expected_optimizer.state[r.A]["exp_avg"])
            compare(p.v_A, expected_optimizer.state[r.A]["exp_avg_sq"])
            compare(p.m_B, expected_optimizer.state[r.B]["exp_avg"])
            compare(p.v_B, expected_optimizer.state[r.B]["exp_avg_sq"])
    for b in adapter.blocks:
        for key, value in b.host_weights.items():
            assert torch.equal(value, original[f"{b.index}/{key}"])
    slot.close(); adapter.close(); reference_adapter.close()


def test_checkpoint_resume_exactly_matches_uninterrupted(tiny, tmp_path):
    config = TrainConfig(str(tiny), str(tiny / "tokens.jsonl"), str(tmp_path / "full"), steps=3, accumulation=2)
    full = run(config)
    cancelled = run(dataclasses.replace(config, output=str(tmp_path / "cancel")), cancel_after_step=1)
    assert cancelled["status"] == "cancelled"
    assert not (tmp_path / "cancel/adapter").exists()
    resumed = run(dataclasses.replace(config, output=str(tmp_path / "resumed")), resume=tmp_path / "cancel/checkpoint-1")
    assert full["status"] == resumed["status"] == "completed"
    before = load_file(str(tmp_path / "full/checkpoint-3/state.safetensors"))
    after = load_file(str(tmp_path / "resumed/checkpoint-3/state.safetensors"))
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert full["metrics"][-1]["loss"] == resumed["metrics"][-1]["loss"]
    artifact = load_file(str(tmp_path / "resumed/peft-adapter/adapter_model.safetensors"))
    assert len(artifact) == 16  # two full (3 each), two linear (1 each), A+B


def test_corruption_rejected_without_mutating_live_state(tiny, tmp_path):
    adapter = Qwen35Adapter(tiny)
    projections = flatten(adapter)
    optimizer = LoRAOptimizer()
    binding = {"config": {"accumulation": 1}, "model": adapter.identity}
    target = tmp_path / "checkpoint"
    save_checkpoint(target, projections, optimizer, binding, 0)
    original = {key: {name: getattr(p, name).clone() for name in FIELDS} for key, p in projections.items()}
    cpu_rng = torch.get_rng_state().clone()
    (target / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(target, projections, optimizer, binding)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert optimizer.step_count == 0
    for key, p in projections.items():
        assert all(torch.equal(getattr(p, name), value) for name, value in original[key].items())
    adapter.close()


@pytest.mark.parametrize("layers", [1, 2, 6])
def test_layer_count_comes_from_checkpoint(tmp_path, layers):
    path = make_tiny(tmp_path / "model", layers=layers)
    adapter = Qwen35Adapter(path)
    assert len(adapter.blocks) == layers
    adapter.close()


@pytest.mark.parametrize("change", [{"model_type": "llama"}, {"tie_word_embeddings": True},
                                   {"attention_dropout": 0.1}, {"attention_bias": True}])
def test_unsupported_architecture_and_capabilities_fail_explicitly(tiny, change):
    path = tiny / "config.json"
    config = json.loads(path.read_text()); config.update(change); path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unsupported"):
        Qwen35Adapter(tiny)


def test_budget_rejected_before_tensor_payload_load(tiny):
    assert inspect_model(tiny)["layers"] == 4
    with pytest.raises(MemoryError, match="resource_budget"):
        Qwen35Adapter(tiny, host_budget_bytes=1)


def test_slot_layout_and_lifecycle_are_serial(tiny):
    adapter = Qwen35Adapter(tiny)
    slot = session(adapter)
    with slot.run():
        with pytest.raises(RuntimeError, match="concurrent"):
            with slot.run():
                pass
        views = slot.load(adapter.blocks[0])
        for key, value in views.items():
            assert value.data_ptr() % 256 == 0
            assert torch.equal(value, adapter.blocks[0].host_weights[key])
        with pytest.raises(RuntimeError, match="active"):
            slot.close()
    slot.close(); slot.close()
    with pytest.raises(RuntimeError, match="closed"):
        slot.load(adapter.blocks[0])
    adapter.close()


def test_generic_engine_accepts_non_qwen_plan_and_registered_factory(tmp_path):
    """Protocol test only: toy math does not count as another supported model."""
    from ram_vram_finetune.adapters.protocol import BlockPlan
    from ram_vram_finetune.adapters.registry import register
    from ram_vram_finetune.finetune.lora import LoRAProjection

    class ToyAdapter:
        def __init__(self, path, *, rank, alpha, device, **_):
            self.device = torch.device(device)
            self.embedding, self.head = torch.randn(11, 6), torch.randn(11, 6)
            a, b = torch.randn(rank, 6) * 0.01, torch.zeros(6, rank)
            projection = LoRAProjection("custom_projection", rank, alpha, 6, 6, a, b,
                torch.zeros_like(a), torch.zeros_like(b), torch.zeros_like(a), torch.zeros_like(a),
                torch.zeros_like(b), torch.zeros_like(b))
            def functional(hidden_states, position_embeddings, attention_mask, weight, custom_lora):
                return hidden_states + torch.tanh(custom_lora.forward(hidden_states, weight))
            self.blocks = [BlockPlan(0, "arbitrary_block_kind", functional,
                                    {"custom_lora": projection}, {"weight": torch.eye(6)},
                                    {"custom_projection": projection})]
            self.identity = "toy-protocol-only"

        def terminal_norm(self, hidden):
            return hidden

        def positions_and_masks(self, valid):
            return (), {"arbitrary_block_kind": None}

        def artifact_key(self, layer, target, parameter):
            return f"toy.{layer}.{target}.{parameter}"

        def runtime_metadata(self):
            return {"architecture": "toy_protocol_only"}

        def close(self):
            self.blocks.clear()

    register("toy_test_only", ToyAdapter)
    dataset = tmp_path / "tokens.jsonl"
    dataset.write_text(json.dumps({"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}) + "\n")
    result = run(TrainConfig(str(tmp_path), str(dataset), str(tmp_path / "run"), adapter="toy_test_only"))
    assert result["optimizer_steps"] == 2
    assert result["adapter_runtime"]["architecture"] == "toy_protocol_only"


def test_unknown_adapter_rejected_without_output_directory(tmp_path):
    config = TrainConfig(str(tmp_path), str(tmp_path / "unused"), str(tmp_path / "output"), adapter="unknown")
    with pytest.raises(ValueError, match="unsupported_architecture"):
        run(config)
    assert not (tmp_path / "output").exists()


def test_checkpoint_existing_target_never_overwritten_and_lock_cleaned(tiny, tmp_path):
    adapter = Qwen35Adapter(tiny)
    optimizer = LoRAOptimizer()
    target = tmp_path / "checkpoint"
    save_checkpoint(target, flatten(adapter), optimizer, {"config": {"accumulation": 1}}, 0)
    before = (target / "state.safetensors").read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(target, flatten(adapter), optimizer, {"config": {"accumulation": 1}}, 0)
    assert before == (target / "state.safetensors").read_bytes()
    assert not list(tmp_path.glob(".stage-*"))
    assert not (tmp_path / ".checkpoint.lock").exists()
    adapter.close()


def test_local_tokenizer_assistant_only_training(tiny, tmp_path):
    from ram_vram_finetune.training.engine import load_rows
    config = TrainConfig(str(tiny), str(tiny / "chat.jsonl"), str(tmp_path / "chat-run"))
    rows = load_rows(config)
    ids, labels = rows[0]
    assert (labels[:, :7] == -100).all()  # entire user turn and assistant prefix
    assert (labels != -100).sum() == 3  # assistant body plus terminal control
    assert torch.equal(ids[labels != -100], labels[labels != -100])
    report = run(config)
    assert report["status"] == "completed" and report["optimizer_steps"] == 2


def test_explicit_cuda_request_does_not_fall_back_to_cpu(tiny, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicit CUDA requested"):
        Qwen35Adapter(tiny, device="cuda:0")


def test_checkpoint_write_failure_leaves_no_partial_publication(tiny, tmp_path, monkeypatch):
    import ram_vram_finetune.checkpoint_io.state as state
    adapter = Qwen35Adapter(tiny)
    def fail(*args, **kwargs):
        raise OSError("injected save failure")
    monkeypatch.setattr(state, "save_file", fail)
    with pytest.raises(OSError, match="injected"):
        save_checkpoint(tmp_path / "checkpoint", flatten(adapter), LoRAOptimizer(), {}, 0)
    assert not (tmp_path / "checkpoint").exists()
    assert not list(tmp_path.glob(".stage-*"))
    assert not (tmp_path / ".checkpoint.lock").exists()
    adapter.close()


def test_run_scoped_reload_capability_accepts_worker_and_rejects_stale_run(tiny):
    from concurrent.futures import ThreadPoolExecutor
    adapter = Qwen35Adapter(tiny)
    slot = session(adapter)
    with slot.run():
        reloader = slot.reloader(adapter.blocks[0])
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(reloader).result()
            with pytest.raises(RuntimeError, match="active session owner"):
                executor.submit(slot.load, adapter.blocks[0]).result()
    with slot.run():
        with pytest.raises(RuntimeError, match="active session owner"):
            reloader()
    slot.close(); adapter.close()


def test_cancellation_during_completion_observer_prevents_adapter_publication(tiny, tmp_path):
    import signal
    def callback(phase, _):
        if phase == "training_complete":
            signal.raise_signal(signal.SIGTERM)
    output = tmp_path / "completion-cancel"
    result = run(TrainConfig(str(tiny), str(tiny/'tokens.jsonl'), str(output)), progress_callback=callback)
    assert result['status'] == 'cancelled' and result['optimizer_steps'] == 2
    assert (output/'checkpoint-2/COMPLETE').exists()
    assert not (output/'adapter').exists() and not (output/'peft-adapter').exists()
