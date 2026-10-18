"""Safe-tensor checkpoints, complete validation before restore, atomic publish."""
from ..buffers import contract_bindings
from ..platform_support import sync_directory as _fsync_dir
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FIELDS = ("A", "B", "grad_A", "grad_B", "m_A", "v_A", "m_B", "v_B")


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(tensors):
    # One independent CPU copy per tensor, including CPU inputs. In particular
    # do not materialize a CUDA->CPU copy and then clone that second time.
    return {k: v.detach().to(device='cpu', copy=True, memory_format=torch.contiguous_format)
            for k, v in tensors.items()}


def _publish(target, tensors, metadata):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / ("." + target.name + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temp = None
    try:
        if target.exists():
            raise FileExistsError(target)
        temp = Path(tempfile.mkdtemp(prefix=".stage-", dir=target.parent))
        snapshot = _snapshot(tensors)
        save_file(snapshot, str(temp / "state.safetensors"))
        del snapshot
        (temp / "metadata.json").write_text(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        manifest = {name: _sha(temp / name) for name in ("state.safetensors", "metadata.json")}
        (temp / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        (temp / "COMPLETE").write_text("complete\n", encoding="utf-8", newline="\n")
        for file in temp.iterdir():
            with file.open("r+b") as handle:
                os.fsync(handle.fileno())
        _fsync_dir(temp)
        os.rename(temp, target)
        temp = None
        _fsync_dir(target.parent)
    finally:
        if temp is not None:
            shutil.rmtree(temp)
        os.close(fd)
        lock.unlink()


def _read(target):
    target = Path(target)
    if (target / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
        raise ValueError("checkpoint incomplete")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != {"state.safetensors", "metadata.json"}:
        raise ValueError("invalid checkpoint manifest")
    for name, digest in manifest.items():
        if _sha(target / name) != digest:
            raise ValueError("checkpoint hash mismatch")
    return load_file(str(target / "state.safetensors")), json.loads((target / "metadata.json").read_text(encoding="utf-8"))


def save_checkpoint(target, projections, optimizer, binding, cursor):
    tensors = {f"{key}/{field}": getattr(p, field) for key, p in projections.items() for field in FIELDS}
    tensors["rng/cpu"] = torch.get_rng_state()
    # Only the selected CUDA device is supported. CPU runs never touch CUDA RNG.
    is_cuda = next(iter(projections.values())).A.device.type == "cuda"
    if is_cuda:
        tensors["rng/cuda"] = torch.cuda.get_rng_state(0)
    metadata = {"format": "rvft.checkpoint.v1", "binding": binding, "optimizer": optimizer.state_dict(),
                "cursor": cursor, "python_rng": random.getstate(), "cuda_rng": is_cuda}
    _publish(target, tensors, metadata)


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def load_checkpoint(target, projections, optimizer, binding):
    tensors, metadata = _read(target)
    if metadata.get("format") != "rvft.checkpoint.v1" or metadata.get("binding") != binding:
        raise ValueError("checkpoint binding/format mismatch")
    expected = {f"{key}/{field}" for key in projections for field in FIELDS} | {"rng/cpu"}
    is_cuda = next(iter(projections.values())).A.device.type == "cuda"
    if is_cuda:
        expected.add("rng/cuda")
    if set(tensors) != expected or metadata.get("cuda_rng") != is_cuda:
        raise ValueError("checkpoint tensor/device inventory mismatch")
    staged = []
    for key, p in projections.items():
        for field in FIELDS:
            old, new = getattr(p, field), tensors[f"{key}/{field}"]
            if old.shape != new.shape or old.dtype != new.dtype or not torch.isfinite(new).all():
                raise ValueError("checkpoint tensor shape/dtype/value mismatch")
            if field.startswith("v_") and (new < 0).any():
                raise ValueError("negative second moment")
            staged.append((old, new.to(old.device)))
    validated_optimizer = type(optimizer)()
    validated_optimizer.load_state_dict(metadata["optimizer"])
    if type(metadata["cursor"]) is not int or metadata["cursor"] < 0:
        raise ValueError("invalid checkpoint cursor")
    if metadata["cursor"] != validated_optimizer.step_count * binding["config"]["accumulation"]:
        raise ValueError("checkpoint accumulation cursor mismatch")
    cpu_rng = tensors["rng/cpu"]
    torch.Generator(device="cpu").set_state(cpu_rng)
    py_rng = _tuples(metadata["python_rng"])
    random.Random().setstate(py_rng)
    if is_cuda:
        torch.Generator(device="cuda:0").set_state(tensors["rng/cuda"])
    with torch.no_grad():
        for old, new in staged:
            old.copy_(new)
    optimizer.load_state_dict(metadata["optimizer"])
    torch.set_rng_state(cpu_rng); random.setstate(py_rng)
    if is_cuda:
        torch.cuda.set_rng_state(tensors["rng/cuda"], 0)
    return metadata["cursor"]


def export_adapter(target, adapter, config):
    tensors = {adapter.artifact_key(b.index, name, field): getattr(p, field)
               for b in adapter.blocks for name, p in b.adapters.items() for field in ("A", "B")}
    metadata = {"format": "rvft.adapter.v1", "model_identity": adapter.identity,
                "rank": config.rank, "alpha": config.alpha, "tensor_keys": sorted(tensors)}
    if getattr(adapter,'storage_metadata',None) is not None:metadata['quantized_base']=adapter.storage_metadata
    typed=contract_bindings(adapter.blocks)
    if typed:metadata['typed_buffers']=typed
    _publish(target, tensors, metadata)
    saved, loaded = _read(target)
    if loaded != metadata or set(saved) != set(tensors) or any(not torch.equal(saved[k], tensors[k].detach().cpu()) for k in saved):
        raise RuntimeError("adapter reload validation failed")
    del saved, loaded
    # PEFT-compatible weight/config companions; private checkpoint state stays separate.
    target = Path(target)
    # Companions are generated in a separate complete directory by export_peft.
    return export_peft(target.parent / "peft-adapter", adapter, config)


def export_peft(target, adapter, config):
    tensors = {adapter.artifact_key(b.index, name, field): getattr(p, field)
               for b in adapter.blocks for name, p in b.adapters.items() for field in ("A", "B")}
    target = Path(target)
    lock = target.parent / ("." + target.name + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temp = None
    try:
        if target.exists():
            raise FileExistsError(target)
        temp = Path(tempfile.mkdtemp(prefix=".peft-", dir=target.parent))
        snapshot = _snapshot(tensors)
        save_file(snapshot, str(temp / "adapter_model.safetensors"))
        del snapshot
        value = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": config.rank,
                 "lora_alpha": config.alpha, "lora_dropout": 0.0, "bias": "none",
                 "target_modules": sorted({name for b in adapter.blocks for name in b.adapters}),
                 "base_model_name_or_path": str(Path(config.model).resolve()), "inference_mode": True}
        (temp / "adapter_config.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
        if getattr(adapter,'storage_metadata',None) is not None:
            (temp/'rvft_quantized_base.json').write_text(json.dumps({
                'format':'rvft.quantized_adapter_base.v1','model_identity':adapter.identity,
                'storage':adapter.storage_metadata,'requires_matching_quantized_base':True,
                'plain_peft_load_on_original_bf16_base_is_not_equivalent':True},indent=2)+'\n', encoding="utf-8", newline="\n")
        typed=contract_bindings(adapter.blocks)
        if typed:
            (temp/'rvft_typed_buffers.json').write_text(json.dumps({
                'format':'rvft.typed_buffer_adapter_base.v1','model_identity':adapter.identity,
                'typed_buffers':typed,'requires_matching_interpretation_and_base':True},indent=2)+'\n', encoding="utf-8", newline="\n")
        loaded = load_file(str(temp / "adapter_model.safetensors"))
        try:
            if set(loaded) != set(tensors) or any(not torch.equal(loaded[k], tensors[k].detach().cpu()) for k in loaded):
                raise RuntimeError("PEFT tensor reload failed")
        finally:
            del loaded  # Release validation ownership before publish or cleanup.
        for file in temp.iterdir():
            with file.open("r+b") as handle:
                os.fsync(handle.fileno())
        _fsync_dir(temp); os.rename(temp, target); temp = None; _fsync_dir(target.parent)
        return target
    finally:
        if temp is not None:
            shutil.rmtree(temp)
        os.close(fd); lock.unlink()
