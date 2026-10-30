# SpillTune — V0.0.1 Fire up

Layer-streaming LoRA fine-tuning with frozen model weights buffered in RAM and loaded into bounded VRAM slots for forward and backward recomputation. This includes training gradients, accumulation, optimizer state and resumable checkpoints. It is not full-parameter training.

The `spilltune` distribution, Python module entry and Linux launcher expose the same `spilltune` commands. Python, PyTorch, CUDA and other dependencies remain external. No model weights, tokenizer payloads, training datasets or third-party environments are included.

## Install and run

Use a compatible Python environment, then install this source with `python -m pip install .` or a built wheel. That normal pip command can resolve external dependencies; use the offline commands below when dependencies are already provisioned. The `models` extra pins Transformers5.13.0; the exact available [CPU lock](requirements/cpu-linux-x86_64-cp314.lock) targets Linux x86_64 CPython3.14 only. See [installation](docs/public/installation.md).

```bash
python -m pip install --no-index --no-deps --no-build-isolation .
spilltune --version
spilltune make-tiny --adapter llama --layers 2 --output tiny-model
spilltune inspect --adapter llama --model tiny-model
```

`make-tiny` creates random tensors and synthetic examples locally, with no download. It does not establish model quality. Set CPU/device, adapter, model/data/output paths explicitly in a JSON [configuration](examples/llama.json), then run `spilltune train --config FILE`. `spilltune resume --config FILE --checkpoint DIR` restores validated matching state. Paths in examples are placeholders to adapt to user-owned directories.

For direct source use before installation: `PYTHONPATH=src python -m ram_vram_finetune.cli --version`. A source checkout and installed wheel require the same external dependencies. The Linux deb creates `/usr/bin/spilltune`; `spilltune --python /absolute/environment/bin/python --check` selects and inspects that external environment without installing it. See [Linux packaging](docs/public/platform-delivery.md).

## Supported scope

Explicit adapters cover declared subsets of Qwen3.5 text, Llama, Qwen2, Qwen3, Qwen3-MoE, DeepSeek-V3, gpt-oss, GLM-4-MoE-Lite and Qwen3.5-MoE. Unsupported schemas, objectives, aliases and formats fail explicitly. See [capabilities](docs/public/capabilities.md), [adapter protocol](docs/public/adapters.md) and [architecture](docs/public/architecture.md).

Default transfers are synchronous. Optional CUDA raw-weight batching, one-layer prefetch and a fixed-size cache are capability-gated; they are not supported for every adapter/storage path. Internal group128 quantization is lossy compressed storage with BF16 computation, not native low-bit training GEMM or GGUF. All frozen weights must fit host memory. Budgets check necessary capacities, not total process peaks; lower VRAM alone does not prove a speed improvement. See [measurement](docs/public/performance.md).

Linux is the current validation platform. Native Windows/MSI is unvalidated. The complete CPU suite and [repeatable checks](docs/public/testing.md) are included; execution scope must be taken from the corresponding validation receipt. No hosted CI result or broad model-quality result is implied by this source release.

## R112 Q4 benchmark

The public [SpillTune-Benchmark R112 report](https://github.com/oldcitynight/SpillTune-Benchmark/blob/main/reports/R112_REPORT.md) includes the sanitized run records, source slice, protocol, checksums and reproducible summary.

| Measure | `spilltune_streaming` | `independent_native_resident` |
| --- | ---: | ---: |
| Sampled whole-device SMI peak | 2.03 GiB | 3.67 GiB |
| Training time, including checkpoint saves | 76.27 min | 60.98 min |
| Paired evaluation outputs and 500-step loss sequence | All 500 outputs and loss values match exactly | Same paired result |

In this single local run, streaming used **44.7% less sampled whole-device SMI peak** and took **25.1% longer**; the measured times remained in the same order of magnitude. Text, token IDs and stop reasons matched for all 500 paired outputs. Stored judgment scores differed by at most 0.6 percentage points across channels despite identical outputs. The comparator was native NF4 resident on CUDA with activation checkpointing; this was **not** a GPU-offload or CPU comparison. SMI is a sampled whole-device peak, not process-exclusive allocated memory. These observations do not establish statistical equivalence, universal quality preservation or performance across hardware.

## License and contribution

Project terms remain **GPL-3.0-or-later** as scoped in [LICENSE.md](LICENSE.md), with unchanged [COPYING](COPYING). Retained [NOTICE](NOTICE), [source attribution](SOURCE_ATTRIBUTION.md) and [third-party notices](THIRD_PARTY_NOTICES.md) accompany the source. No additional linking exception is granted. External runtimes, models, data and outputs retain their applicable terms. See [contributing](CONTRIBUTING.md) and [release](docs/public/release.md).

## Authors

Primary author credit: **ChatGPT-Codex <no-reply@openai.com>**. Secondary author and maintainer: **oldcitynight <nightoldcity@gmail.com>**. SpillTune was developed through AI collaboration under human direction. These are the user-designated project credits; this is not an official OpenAI project, endorsement or signature. Credit does not assign legal copyright ownership to an AI system or transfer upstream rights. Retained source attribution and license notices remain authoritative.

The internal Python namespace remains `ram_vram_finetune` for import and checkpoint compatibility; install the `spilltune` distribution and use the `spilltune` command. Internal schema identifiers and adapter configuration keys remain stable.
