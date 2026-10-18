# Adding an adapter

Implement a strict factory and inspector; do not infer support from a model name. A factory accepts `path, *, rank, alpha, device, host_budget_bytes, device_budget_bytes` and, when applicable, explicit `quantization`/`quantized_compute` options. Register it with `register(name, factory, inspector=...)`; configuration cannot import arbitrary remote code.

`ModelAdapter` exposes frozen CPU `embedding`, frozen selected-device `head`, `device`, a content-bound `identity`, ordered `blocks`, and these methods:

```python
positions_and_masks(valid)  # position embeddings plus {block_kind: mask}
terminal_norm(hidden)
artifact_key(layer, target, parameter)
runtime_metadata()
close()
```

Each `BlockPlan(index, kind, functional, kwargs, host_weights, adapters, storage_specs=None)` has one pure computation entry:

```python
functional(hidden_states, position_embeddings, attention_mask, **weights, **kwargs)
```

Host tensors are frozen, contiguous and on CPU. A slot view lives only until the next reload. The function returns independent hidden states and must not capture old slot tensors. Forward and backward recomputation use identical semantics. Projection objects occur directly and once in `kwargs`, match `adapters` by identity, and own independent FP32 A/B, gradients and moments.

Validate config, complete tensor inventory, boundary shapes/dtypes, layer schedule, tied-weight semantics, cache/dropout, position and objective policy. Where metadata shape checks are incomplete, fail before computation and document the loading boundary; never substitute a synthetic model, skip layers, silently untie weights or switch architecture. Record aliases, source layout, backend, excluded non-text/MTP roles and compressed-base policy.

An adapter's identity binds effective configuration and all used physical tensors by name/dtype/shape/content. Alias sources are canonical and read once. Preserve existing identity conventions for backward-compatible defaults; introducing a default field into hashed configuration can invalidate historical checkpoints.

For compression, declare eligible matrix and FP32-state keys. A `Record` maps a logical tensor to raw or compressed parts; `storage_specs` maps block weights to those parts. Verify decoded shapes/bytes, exact recipes and matching-base identity. Do not retain an undisclosed full-precision decoder. A format name resembling a GGUF type does not establish compatibility.

## Minimum validation

- Random native architecture with each supported layer kind, FP32 and applicable BF16, zero/nonzero B and distinguishing norms/biases/router behavior.
- Independent native whole-model positions/masks/RoPE and loss, input and every A/B gradient. Copy LoRA initial values only; do not share trainables or the base computation graph.
- Unequal supervised-token accumulation and two updates with parameters and all moments. Describe any same-gradient low-precision optimizer protocol separately from an independent trajectory.
- Frozen bases, declared reload order, unsupported config/layout/tie/objective failures and budget/lifecycle/cancellation behavior.
- Complete state, RNG/cursor/optimizer and exports after a fresh-process restore. Perturb state before loading and check restoration immediately.
- CPU checks first. GPU/real-source acceptance needs a separate bounded resource plan, small reproducible inputs, swap/RSS and allocator peaks, transfer/time accounting and explicit scope.

The registry protocol is model-neutral. Existing adapters implement limited subsets; their success is not an automatic promise for another architecture or derivative checkpoint.

## Explicit boundaries

Common full/linear attention kinds, algorithms and operators may remain shared.
They do not determine concrete model module paths. Call
`create_lora_for_block(..., targets=ordered_canonical_paths)` explicitly;
missing, duplicate and invalid paths fail. Leaf aliases resolve uniquely within
that instance. Legacy adapters explicitly use mappings in `adapters.targets`.
The previous low-level implicit-target call is no longer accepted.

An inspector explicitly declares `quantized_export` as
`{"version": 1, "formats": [...], "rejection_reason": "..."}`. Missing or
malformed declarations deny export before creating output or reading tensor
payloads. This does not grant training/compute support for an unvalidated format.

For reviewed source formats outside the existing archive recipe, a raw
`BlockPlan.buffer_contract = RawBufferContract(decoder_id, identity,
decode_workspace_bytes)` permits true typed parts (including U8/I32) without
reinterpreting them as BF16. Identity is a SHA256 binding of interpretation/layout
semantics, in addition to the adapter's content-bound base identity. The existing
functional callable interprets the raw parts; the common layer only transfers
and lends them. The workspace declaration must bound the maximum live decoding
intermediates in forward and backward recomputation. It is a budget reservation,
not allocated scratch, and does not defend against a callback that lies.

New contracts use synchronous transfers only. CPU budget includes host parts,
slot/upload storage, known alignment overhead and declared decoding workspace;
device budget also reserves the workspace. Session metadata is deeply snapshotted
and mutation is rejected. Checkpoints additionally bind interpretation, layout
and workspace. Exports carry matching typed-buffer metadata; PEFT weights alone
do not establish matching-base semantics. Legacy archive schemas/identities
remain unchanged. This extension does not implement MXFP4 or another model family.
