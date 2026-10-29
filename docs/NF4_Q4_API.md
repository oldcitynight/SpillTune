# Qwen3.5 NF4 streaming API

Install the optional runtime with `pip install 'spilltune[nf4]'` and the model framework with the pinned Transformers version. Supply a local Qwen3.5 text checkpoint and a tokenized dataset; SpillTune does not download model weights.

For a full-size source checkpoint, build its canonical artifact in a bounded job first:

```python
from ram_vram_finetune.quantization.nf4_model_builder import (
    build_profiled_qwen35_nf4_artifact,
)
from ram_vram_finetune.quantization.nf4_model_state import save_nf4_model

artifact = build_profiled_qwen35_nf4_artifact(
    model_dir,
    limits={
        "max_source_bytes": source_limit,
        "max_source_tensor_bytes": tensor_limit,
        "max_canonical_payload_bytes": artifact_limit,
    },
    conversion_guard=your_resource_guard,
)
save_nf4_model(artifact_dir, artifact)
```

The geometry limits are explicit admission bounds; they are not measured peak-memory guarantees. The conversion guard must enforce the host resource policy selected by the operator.

Train from an already-built artifact and pretokenized rows:

```python
from ram_vram_finetune.quantization.nf4_model_state import load_nf4_model
from ram_vram_finetune.training.nf4_trainer import train_qwen35_nf4_streaming

artifact = load_nf4_model(artifact_dir)
result = train_qwen35_nf4_streaming(
    artifact,
    rows_by_id={
        "row-1": {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        "row-2": {"input_ids": [1, 4, 5], "labels": [-100, 4, 5]},
    },
    order_ids=["row-1", "row-2"],
    updates=1,
    accumulation=2,
    output_path="adapter.safetensors",
)
```

The training API uses the streaming CUDA adapter explicitly, with BF16 frozen compute and FP32 LoRA factors. It reports per-update loss and writes a final adapter-only safetensors file. The file carries the complete source-bound runtime profile (including rank, alpha, seed, targets, and quantization settings), model/artifact binding, tensor inventory, and training-data/order hashes. The built-in checkpoint loader does not load this final-adapter format; use the explicit adapter loader below.

Load the saved adapter in a fresh process by reading its profile, constructing the same model configuration, and asking the adapter loader to validate the artifact identity, target inventory, and tensor shapes before copying weights:

```python
from ram_vram_finetune.adapters.qwen35.nf4_cuda import NF4Qwen35CUDA
from ram_vram_finetune.quantization.nf4_model_state import load_nf4_model
from ram_vram_finetune.training.nf4_trainer import (
    load_nf4_adapter,
    read_nf4_adapter_metadata,
)

artifact = load_nf4_model(artifact_dir)
metadata = read_nf4_adapter_metadata("adapter.safetensors")
profile = metadata["runtime_profile"]
model = NF4Qwen35CUDA(
    artifact,
    mode="streaming",
    runtime_profile=profile,
    rank=profile["rank"],
    alpha=profile["alpha"],
    seed=profile["seed"],
    device="cuda:0",
)
load_nf4_adapter(model, "adapter.safetensors")
```

The loader validates the complete metadata, model binding, tensor inventory, shapes, dtypes, and finite values before copying any adapter tensor. These validation errors leave model parameters unchanged. A device-transfer/runtime failure during the subsequent per-parameter copy can still leave a partially loaded model; if loading raises during that copy, discard the model instance and construct a fresh one before retrying. The adapter file is opened once for metadata validation and again for tensor loading, so keep the file unchanged and unreplaced for the entire `load_nf4_adapter` call.

Adapter publication uses an atomic no-replace filesystem link. A second writer targeting an existing path gets `FileExistsError`; it cannot replace the first adapter. The API does not resume from an intermediate optimizer checkpoint. The R112 resume/evaluation harness belongs in SpillTune-Benchmark. Performance and GPU compatibility depend on the installed hardware and runtime and are not guaranteed by installation alone.
