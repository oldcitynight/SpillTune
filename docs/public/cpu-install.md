# Locked CPU installation

This dependency lock covers **Linux x86_64, regular CPython 3.14, and glibc 2.28 or newer**. It pins the complete runtime, model-definition, test, and build dependency closure, including Torch **2.13.0+cpu**, Transformers **5.13.0**, Safetensors **0.8.0**, pytest **9.1.1**, and setuptools **83.0.0**. It contains 37 wheels totaling 244,633,870 bytes (about 233.3 MiB). Other operating systems, architectures, Python versions, free-threaded Python, and CUDA combinations are unvalidated.

The preparation used official package metadata and matching CPython 3.14 wheel tags. The new installation target is system CPython **3.14.4**; historical project evidence used **3.14.6**. A prepared lock is not itself a completed installation or numerical test: use the release validation record for the result of executing these steps.

Run the following commands from this source candidate's root. Use a Python 3.14 interpreter installed independently of any old project or Conda environment. The helper uses only Python's standard library and does not need the project installed.

## Obtain and verify the exact wheels

The manifest records one official HTTPS URL, filename, byte count, and SHA256 per wheel. The requirements lock contains only exact package versions and hashes, with no direct URLs. Fetching and offline installation are separate steps.

```sh
python3.14 tools/portable/fetch_cpu_wheels.py fetch \
  --manifest requirements/cpu-linux-x86_64-cp314.manifest.json \
  --lock requirements/cpu-linux-x86_64-cp314.lock \
  --wheelhouse .cpu-wheelhouse \
  --receipt .cpu-wheelhouse/download-ledger.json

python3.14 tools/portable/fetch_cpu_wheels.py verify \
  --manifest requirements/cpu-linux-x86_64-cp314.manifest.json \
  --lock requirements/cpu-linux-x86_64-cp314.lock \
  --wheelhouse .cpu-wheelhouse \
  --receipt .cpu-wheelhouse/wheel-verification.json
```

The helper fetches only the manifest's exact URLs on `download.pytorch.org/whl/cpu`, `pypi.org`, and `files.pythonhosted.org`. It rejects other hosts, cross-host redirects, changed URLs, missing or changed content lengths, source distributions, GPU package names, size mismatches, and SHA256 mismatches. It uses no account credentials, private indexes, inherited proxy settings, package discovery, or automatic retries. It never downloads a model, tokenizer asset, or corpus. The `tokenizers` Python library is a locked dependency; it is distinct from downloading a model's tokenizer files.

A fetch is capped at 1 GiB of package response bytes and 600 seconds, including any metadata usage supplied through `--prior-metadata-bytes` and `--prior-metadata-seconds`. Preparation and validation operators must pass their recorded prior usage when sharing that budget. An interrupted fetch leaves its failure ledger and `.part` file for review. A pre-existing ledger prevents an accidental second run; resolve the failure explicitly before starting a new budgeted attempt.

Verification reads every wheel, checks its size and SHA256, compares its identity and dependency metadata with the manifest, and records hashes of license and notice members. Keep the wheelhouse and receipts as local build inputs; they are excluded from the public source archive.

## Create a clean environment and install offline

```sh
python3.14 -I -m venv .venv-cpu

python3.14 tools/portable/fetch_cpu_wheels.py install \
  --manifest requirements/cpu-linux-x86_64-cp314.manifest.json \
  --lock requirements/cpu-linux-x86_64-cp314.lock \
  --wheelhouse .cpu-wheelhouse \
  --python .venv-cpu/bin/python \
  --receipt .cpu-wheelhouse/offline-install.json
```

The helper requires `include-system-site-packages = false`. The tested system Python supplies bootstrap pip 25.1.1 through its standard-library ensurepip; that bootstrap is recorded separately from the 37-wheel application/build/test lock. It clears inherited `PIP_*`, `PYTHONPATH`, and `PYTHONHOME` settings, disables pip configuration with `PIP_CONFIG_FILE=/dev/null`, verifies every wheel again, and runs pip with `--no-index --find-links --only-binary=:all: --require-hashes --no-cache-dir`, followed by `pip check`. It does not copy an old environment, add a `.pth` bridge, or resolve CUDA dependencies. Its installation receipt records both pip results.

Install this source candidate using the already installed build tools, with network resolution disabled:

```sh
env -i PATH=/usr/bin:/bin HOME="$HOME" PIP_CONFIG_FILE=/dev/null \
  .venv-cpu/bin/python -I -m pip install \
  --no-index --no-deps --no-build-isolation --no-cache-dir .

.venv-cpu/bin/python -I -m pip check
.venv-cpu/bin/python -I -c \
  'import torch; assert torch.__version__ == "2.13.0+cpu"; assert torch.version.cuda is None; print(torch.__version__)'
```

Run `.venv-cpu/bin/python tools/portable/ci_cpu.py --require-clean-cpu --output ../spilltune-clean-cpu-check` to require measured isolated CPU dependency paths and reject CUDA Torch or borrowed system packages before executing the candidate.

Use the offline random example and CPU validation commands in [installation.md](installation.md) and [testing.md](testing.md). Keep `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `CUDA_VISIBLE_DEVICES=""` set for those runs. The dependency lock supplies the Transformers implementation without requiring pretrained assets.

## Scope of license evidence

The manifest preserves dependency source/hash/license metadata for external user provisioning; it conveys no dependency binaries or new permissions. See [LICENSE.md](../../LICENSE.md) and [third-party notices](../../THIRD_PARTY_NOTICES.md).
