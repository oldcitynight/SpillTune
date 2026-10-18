# Linux CLI delivery

V0.0.1 Fire up uses version0.0.1 in package metadata, module/console `--version` and Linux deb. Standard source/wheel and the native Linux launcher invoke the same CLI. The launcher is own C code plus own Python payload, with no embedded interpreter, PyTorch, CUDA or models.

Launcher-only options precede the tool command: `--python ABSOLUTE`, `--app ABSOLUTE`, `--check`; `--` is optional. Interpreter selection is explicit option, SPILLTUNE_PYTHON, active VIRTUAL_ENV, active CONDA_PREFIX, then `/usr/bin/python3`. A selected bad environment fails without silently trying another. The selected Python runs `-I -B -X utf8`; own payload is located relative to the launcher. `--check` imports dependencies and reports paths/ABI/build without initializing CUDA. Import success is not driver/training acceptance.

Build the own wheel using installed setuptools, then build a deb using only existing compiler/readelf/dpkg tools:

```bash
python -c "import setuptools.build_meta as b; print(b.build_wheel('dist'))"
python tools/portable/build_deb.py --wheel dist/spilltune-0.0.1-py3-none-any.whl --output dist/deb --maintainer 'oldcitynight <nightoldcity@gmail.com>'
```

The explicit maintainer is release metadata, not an example address to publish. Output must be new. The current helper targets Linux amd64; libc minimum comes from actual ELF versioned symbols. Depends covers Python3>=3.11 and libc; external Torch/CUDA provisioning is explicit. There are no maintainer scripts, silent downloads, apt-source/driver changes or agreement flags. The package installs `/usr/bin/spilltune`, own share payload, source and notices. Runtime outputs/venvs belong in user-owned paths.

Actual install/configure/remove acceptance must come from a dpkg execution receipt and status/file checks. Unpacking is a smaller test. A write-isolated mount-namespace install is not a booted VM or cross-distro test. Native Windows/MSI is skipped until a native executor/toolchain exists; a py3-none-any tag or platform-parameter tests do not establish Windows execution.

RSS is null where unavailable. Windows directory-fsync is explicitly unsupported; POSIX failures propagate. File flush/close, same-parent staging/rename and cooperative exclusive lock remain. Windows crash durability and native cleanup behavior need native validation.
