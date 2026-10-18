# Contributing

Keep adapter validation, transfer/storage scheduling and fine-tuning math separate. New architectures require explicit registry entries, native-reference gradient/update checks and unsupported-branch rejection; do not add silent architecture/device fallbacks.

Run [CPU checks](docs/public/testing.md) with compatible external dependencies. Review intentional test/inventory changes together, and report measured Linux/CPU/GPU/Windows scope accurately. No remote CI success is implied by a workflow file.

Preserve original source headers, applicable terms and notices. Identify the exact origin/version and modified scope of new adapted material. Do not include credentials, machine configurations, development history, caches, environments, model weights or training datasets. This guide adds no separate contribution agreement or linking exception.
