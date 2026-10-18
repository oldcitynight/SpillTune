# Install V0.0.1 Fire up

Use Python3.11+ syntax with a compatible external runtime. Package requirements declare Torch>=2.6 and Safetensors>=0.5; model adapters pin Transformers5.13.0 through the `models` extra because they use its internals. These lower bounds are not a tested version matrix. The reproducible CPU lock targets Linux x86_64 CPython3.14 with Torch2.13.0+cpu, Transformers5.13.0 and Safetensors0.8.0.

Normal source installation is `python -m pip install '.[models]'`; it may fetch dependencies from the user-configured sources. For offline own-tool installation with compatible dependencies and setuptools already provisioned:

```bash
python -m pip install --no-index --no-deps --no-build-isolation .
spilltune --version
python -m ram_vram_finetune.cli --version
```

The installed console script lives in that Python environment’s bin/Scripts directory. Activate the environment or invoke its absolute entry. A direct source run uses `PYTHONPATH=src python -m ram_vram_finetune.cli ...` from the source root. No old project path is required.

For a reviewed Linux deb, ordinary package installation creates `/usr/bin/spilltune`; it does not install Torch/CUDA. Select a prepared external Python with `spilltune --python /absolute/environment/bin/python --check` or SPILLTUNE_PYTHON. The [launcher/deb guide](platform-delivery.md) describes selection and the [CPU guide](cpu-install.md) describes optional locked provisioning. Native Windows/MSI is unvalidated.

Model, tokenizer, data, checkpoint and output paths must be explicit user-owned paths. Examples contain placeholders, not a bundled model or environment. There is no automatic general environment deployer or llama.cpp integration.
