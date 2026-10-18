# V0.0.1 capabilities

All adapters train projection LoRA with frozen bases and text causal CE. Full bases, routers and experts remain frozen while their input derivatives participate in backward recomputation. See [adapters](adapters.md) for interfaces.

| Adapter | Declared subset and limits |
|---|---|
| qwen35 | Dense text hybrid linear/full schedule; offset RMSNorm, gated Q; frozen canonical tied head only without compression; vision/MTP excluded |
| llama | Dense eager GQA/default RoPE; no projection bias, tied embeddings, scaled/sliding attention |
| qwen2 | Dense GQA and fixed Q/K/V bias; active sliding window/tied embeddings rejected |
| qwen3 | GQA/per-head QK norm/frozen canonical head; raw FP32/BF16, no compression |
| qwen3_moe | Frozen eager router/experts, packed or per-expert layout; attention LoRA; aux disabled; whole layer streamed |
| qwen35_moe | BF16 text hybrid routed/shared experts; optional INT4 storage/BF16 compute; synchronous policy; vision/MTP/aux excluded |
| deepseek_v3 | Declared MLA/grouped-MoE/default or YaRN branches; attention LoRA/frozen experts; no universal source/layout acceptance |
| gpt_oss | Native MXFP4 expert storage decoded to BF16; full/sliding eager attention q/v/o LoRA; token-ID JSONL, synchronous whole-layer path |
| glm4_moe_lite | Firstdense/sparse MLA/shared/routed experts; attention LoRA; optional exact MTP branch validated/hashed/excluded; CE only |

Group128 E4M3FN/E5M2/INT8/INT6/INT4 internal storage is restricted to declared untied Qwen3.5/Llama/Qwen2/DeepSeek subsets; Qwen3.5-MoE has a separate explicit INT4 path. Compression is lossy and decoded for BF16 math. These formats are not GGUF, native low-bit GEMM or a full paged-optimizer QLoRA stack.

Default is synchronous. Optional raw CUDA prefetch/cache and batching require explicit compatible capability/configuration; compressed paths may reject them. No universal asynchronous lifecycle is implied. Arbitrary trainable shared aliases, expert-granularity paging, disk-backed bases, CPU optimizer offload, distributed or full-parameter training are not provided. Unsupported combinations fail rather than switching architecture/device silently.

All frozen weights and the currently loaded dataset must fit host resources. Capacity checks do not bound total process/activation/allocator peaks. Native Windows/MSI and broad quality/throughput claims remain unvalidated. Test/measurement scope belongs to the exact candidate receipt.
