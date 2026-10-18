# Measurement protocol

Measure identical source/data/configuration/seeds and compare actual outputs/gradients before timing alternatives. Record warmup, initialization and steady-state windows separately; synchronize CUDA at timed boundaries. Report wall time/tokens per second, RAM high-water/current RSS, swap, CUDA allocated/reserved peaks, head/slot/LoRA/optimizer capacities and H2D/D2H payload/time.

Lower VRAM consumption is not evidence of greater speed. Cgroup memory peak is not process RSS; unavailable metrics must be null. Async overlap requires a trace that establishes transfer/compute lifetimes and resource peaks. Compression changes base values and needs its own quality assessment.

The included synthetic tiny CPU tests validate implementation paths, not pretrained quality, large-model throughput, long contexts or every device/architecture. This release does not carry private development benchmark history. Native GPU benchmarks must use an explicit bounded per-job plan and the exact candidate/dependency identities.
