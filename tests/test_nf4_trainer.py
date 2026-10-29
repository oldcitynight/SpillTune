from __future__ import annotations

import os
import subprocess
import sys

import torch
import pytest

from ram_vram_finetune.training.nf4_preflight import PreparedTrainingRows
from ram_vram_finetune.quantization.nf4_runtime_profile import make_runtime_profile
from ram_vram_finetune.training.nf4_trainer import (
    load_nf4_adapter,
    read_nf4_adapter_metadata,
    run_nf4_updates,
    save_nf4_adapter,
)


class _FakeStreamingModel:
    execution_arm = "spilltune_streaming"

    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.binding = None
        self._loss_fn = None

    def named_adapter_parameters(self):
        return {"0:linear_attn.out_proj/A": self.weight}

    def checkpoint_binding(self):
        return {"architecture": "test-double", "mode": "streaming"}

    def prepare_preflight_training(self, rows, *, binding, order):
        self.binding = dict(binding)
        self.snapshot = PreparedTrainingRows(
            rows, binding=binding, vocab_size=16, order=order
        )
        self.snapshot.claim(self)

        def loss_fn(row_id, *, normalization, token_chunk):
            self.snapshot.row(self, row_id)
            assert token_chunk == 2
            target = 1.0 if row_id == "a" else 2.0
            return ((self.weight - target) ** 2) / normalization, None

        self._loss_fn = loss_fn
        return loss_fn


def test_run_nf4_updates_uses_preflight_hashes_and_updates_parameters():
    model = _FakeStreamingModel()
    rows = {
        "a": {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        "b": {"input_ids": [1, 4, 5], "labels": [-100, 4, 5]},
    }
    result = run_nf4_updates(
        model, rows, ["a", "b"], updates=1, accumulation=2, learning_rate=1e-2
    )

    assert result["updates"] == 1
    assert result["microbatches"] == 2
    assert len(result["losses"]) == 1
    assert result["losses"][0] > 0
    assert model.weight.item() > 0
    assert len(model.binding["training_rows_sha256"]) == 64
    assert len(model.binding["training_order_sha256"]) == 64


def test_run_nf4_updates_rejects_incomplete_order_before_model_preflight():
    model = _FakeStreamingModel()
    try:
        run_nf4_updates(
            model,
            {"a": {"input_ids": [1, 2], "labels": [-100, 2]}},
            ["a"],
            updates=1,
            accumulation=2,
        )
    except ValueError as exc:
        assert "does not cover" in str(exc)
    else:
        raise AssertionError("incomplete order should be rejected")
    assert model.binding is None


class _AdapterModel:
    def __init__(self, profile):
        self.runtime_profile = dict(profile)
        self.a = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        self.b = torch.nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(4, 2))
        self._binding = {
            "artifact_identity": profile["artifact_identity"],
            "source_identity": profile["source_identity"],
            "record_state_identity": "c" * 64,
            "config_sha256": profile["config_sha256"],
            "initialization_identity": "d" * 64,
            "rank": profile["rank"],
            "alpha": profile["alpha"],
            "targets": list(profile["targets"]),
            "runtime_profile": dict(profile),
        }

    def checkpoint_binding(self):
        return dict(self._binding)

    def named_adapter_parameters(self):
        target = self.runtime_profile["targets"][0]
        return {f"{target}/A": self.a, f"{target}/B": self.b}


def _adapter_profile(artifact_identity="a" * 64):
    config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 16,
        "intermediate_size": 32,
        "vocab_size": 64,
        "num_hidden_layers": 1,
        "layer_types": ["linear_attention"],
        "tie_word_embeddings": True,
    }
    return make_runtime_profile(
        config, artifact_identity, "b" * 64, rank=2, alpha=4.0, seed=17
    )


def _training_metadata():
    return {
        "updates": 3,
        "accumulation": 2,
        "training_rows_sha256": "e" * 64,
        "training_order_sha256": "f" * 64,
    }


def test_nf4_adapter_safetensors_roundtrip_preserves_profile_and_weights(tmp_path):
    profile = _adapter_profile()
    source = _AdapterModel(profile)
    expected = {name: value.detach().clone() for name, value in source.named_adapter_parameters().items()}
    path = save_nf4_adapter(tmp_path / "adapter.safetensors", source, training=_training_metadata())

    metadata = read_nf4_adapter_metadata(path)
    assert metadata["runtime_profile"] == profile
    assert metadata["runtime_profile"]["rank"] == 2
    assert metadata["runtime_profile"]["alpha"] == 4.0
    assert metadata["runtime_profile"]["seed"] == 17
    assert metadata["training"] == _training_metadata()
    assert metadata["parameters"]["0:linear_attn.out_proj/A"] == {
        "shape": [2, 3], "dtype": "float32"
    }

    restored = _AdapterModel(profile)
    restored.a.data.zero_()
    restored.b.data.zero_()
    loaded = load_nf4_adapter(restored, path)
    assert loaded == metadata
    for name, parameter in restored.named_adapter_parameters().items():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)


def test_nf4_adapter_save_is_no_replace_and_loader_rejects_other_artifact(tmp_path):
    profile = _adapter_profile()
    source = _AdapterModel(profile)
    path = save_nf4_adapter(tmp_path / "adapter.safetensors", source, training=_training_metadata())
    original = {name: value.detach().clone() for name, value in source.named_adapter_parameters().items()}
    source.a.data.fill_(99)
    with pytest.raises(FileExistsError):
        save_nf4_adapter(path, source, training=_training_metadata())

    wrong_model = _AdapterModel(_adapter_profile("9" * 64))
    wrong_model.a.data.fill_(77)
    wrong_before = wrong_model.a.detach().clone()
    with pytest.raises(ValueError, match="different model/artifact/profile"):
        load_nf4_adapter(wrong_model, path)
    torch.testing.assert_close(wrong_model.a, wrong_before, rtol=0, atol=0)

    restored = _AdapterModel(profile)
    load_nf4_adapter(restored, path)
    for name, parameter in restored.named_adapter_parameters().items():
        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)


def test_nf4_public_modules_import_without_optional_cuda_runtime_packages():
    script = r"""
import sys
from importlib.abc import MetaPathFinder

class DenyOptionalRuntime(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bitsandbytes" or fullname.startswith("bitsandbytes."):
            raise AssertionError("bitsandbytes was imported during public module import")
        if fullname == "transformers" or fullname.startswith("transformers."):
            raise AssertionError("transformers was imported during public module import")

sys.meta_path.insert(0, DenyOptionalRuntime())
from ram_vram_finetune.adapters.qwen35.nf4_cuda import NF4Qwen35CUDA
from ram_vram_finetune.adapters.qwen35.nf4_cpu import NF4Qwen35CPU
from ram_vram_finetune.quantization.nf4_backend_cuda import backend_info_cuda
from ram_vram_finetune.quantization.nf4_model_builder import build_profiled_qwen35_nf4_artifact
from ram_vram_finetune.training.nf4_cuda_pool import CUDANF4LeasePool
from ram_vram_finetune.training.nf4_cuda_recompute import ControlledNF4CUDADecoder
from ram_vram_finetune.training.nf4_model_recompute import ControlledNF4Decoder
from ram_vram_finetune.training.nf4_trainer import train_qwen35_nf4_streaming
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
