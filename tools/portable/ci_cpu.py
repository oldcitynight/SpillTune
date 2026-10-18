"""Offline own-package build/install and inventory-audited CPU verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile
import tarfile
import time
import tomllib
import zipfile

from suite_audit import audit_result, clean_environment, source_snapshot, validate_inventory, validate_import_paths
from cpu_environment import collect_environment_snapshot, validate_clean_cpu

ROOT = Path(__file__).resolve().parents[2]


def console_entry_command(installed, *, platform_name=None):
    name = sys.platform if platform_name is None else platform_name
    if name == 'win32':
        return [str(PureWindowsPath(installed) / 'Scripts' / 'spilltune.exe')]
    return [sys.executable, str(Path(installed) / 'bin' / 'spilltune')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize_audit(inventory, ledger, exit_code, *, collect_only, installed_root=None):
    """Only the parent, after pytest exits, can accept its complete event ledger."""
    if ledger.get("status") != "RECORDED" or ledger.get("mode") != ("collection" if collect_only else "execution"):
        raise ValueError("missing or malformed pytest event ledger")
    result = audit_result(inventory, ledger["collected_nodeids"], ledger["reports"],
                          collect_only=collect_only, exit_code=exit_code,
                          deselected=ledger["deselected"], collection_problems=ledger["collection_problems"])
    if ledger.get("validation_status") != "PASS":
        result["status"] = "FAIL"
        result["errors"].append("provisional_ledger_validation_failed")
    if installed_root is not None:
        try:
            if ledger.get("installed_root") != str(Path(installed_root).resolve()):
                raise ValueError("installed root differs from requested target")
            validate_import_paths(ledger.get("imported_modules", {}), installed_root)
        except (ValueError, TypeError) as exc:
            result["status"] = "FAIL"
            result["errors"].append(str(exc))
    return result



def read_source_inventory(root):
    root = Path(root)
    content = (root / "tests/cpu-inventory.json").read_bytes()
    inventory = validate_inventory(json.loads(content))
    if source_snapshot(root) != inventory["source_files"]:
        raise ValueError("input test source differs from reviewed inventory; refresh deliberately and update allowlist")
    return inventory, hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-clean-cpu", action="store_true", help="refuse borrowed dependencies, system sites, or CUDA Torch")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = clean_environment()
    commands = []
    started = time.monotonic()

    def call(label, command, cwd=None, required=True):
        t = time.monotonic()
        result = subprocess.run([str(x) for x in command], cwd=Path(tempfile.gettempdir()) if cwd is None else cwd, env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=180)
        (out / (label + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8", newline="\n")
        commands.append({"label": label, "exit_code": result.returncode,
                         "seconds": time.monotonic() - t})
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(label, result.returncode, flush=True)
        if required and result.returncode:
            raise RuntimeError(label + " failed; inspect private local log")
        return result

    def runtime_identity(source, installed, wheel):
        expected = {p.relative_to(source / "src").as_posix(): sha(p)
                    for p in (source / "src/ram_vram_finetune").rglob("*.py")}
        with zipfile.ZipFile(wheel) as archive:
            actual = {name: hashlib.sha256(archive.read(name)).hexdigest()
                      for name in archive.namelist() if name.startswith("ram_vram_finetune/") and name.endswith(".py")}
        if not expected or actual != expected or any(sha(installed / name) != digest for name, digest in expected.items()):
            raise ValueError("source/wheel/installed runtime identity mismatch")
        return expected

    try:
        dependency_environment = collect_environment_snapshot()
        if args.require_clean_cpu:
            validate_clean_cpu(dependency_environment)
        (out / "dependency-environment.json").write_text(json.dumps(dependency_environment, indent=2) + "\n", encoding="utf-8", newline="\n")
        input_inventory, input_inventory_sha = read_source_inventory(ROOT)
        policy = json.loads((ROOT / "release-allowlist.json").read_text(encoding="utf-8"))
        source = out / "build-source"
        source.mkdir()
        if len(set(policy["files"])) != len(policy["files"]):
            raise ValueError("duplicate allowlist file")
        for rel in policy["files"]:
            p = ROOT / rel
            if not p.is_file() or p.is_symlink() or Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError("invalid allowlisted source")
            target = source / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        call("candidate-scan", [sys.executable, source / "tools/portable/check_candidate.py", "--root", source])
        call("license-materials", [sys.executable, source / "tools/portable/check_licenses.py", "--root", source, "--output", out / "license-materials.json"])
        inventory_path = source / "tests/cpu-inventory.json"
        inventory_content = inventory_path.read_bytes()
        inventory_sha = hashlib.sha256(inventory_content).hexdigest()
        inventory = validate_inventory(json.loads(inventory_content))
        if inventory_sha != input_inventory_sha or inventory != input_inventory:
            raise ValueError("staged inventory differs from original input")
        if source_snapshot(source) != inventory["source_files"]:
            raise ValueError("test source differs from reviewed inventory")
        version = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        call("build-wheel", [sys.executable, "-c", "import setuptools.build_meta as b; print(b.build_wheel('dist'))"], cwd=source)
        call("build-sdist", [sys.executable, "-c", "import setuptools.build_meta as b; print(b.build_sdist('dist'))"], cwd=source)
        wheel = source / f"dist/spilltune-{version}-py3-none-any.whl"
        sdist = source / f"dist/spilltune-{version}.tar.gz"
        with tarfile.open(sdist) as archive:
            names = {"/".join(Path(member.name).parts[1:]) for member in archive.getmembers() if member.isfile()}
        if not set(policy["files"]) <= names:
            raise ValueError("sdist omitted allowlisted source or test inventory")
        call("package-license-materials", [sys.executable, source / "tools/portable/check_licenses.py", "--root", source, "--wheel", wheel, "--sdist", sdist, "--output", out / "package-license-materials.json"])
        installed = out / "installed"
        call("install-project-wheel", [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--target", installed, wheel])
        identity = runtime_identity(source, installed, wheel)
        env["PYTHONPATH"] = str(installed) + os.pathsep + str(source / "tools/portable")
        import_code = ("import sys,json,pathlib,ram_vram_finetune as p; from ram_vram_finetune.training import engine; "
                       f"assert p.__version__=={version!r}; assert pathlib.Path(p.__file__).resolve().is_relative_to(pathlib.Path({str(installed)!r}).resolve()); "
                       "assert not any(k=='backend' or k.startswith('backend.') or k=='streaming' or k.startswith('streaming.') for k in sys.modules); "
                       "assert all(not getattr(m,'__file__',None) or pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(" + repr(str(installed)) + ").resolve()) "
                       "for n,m in sys.modules.items() if n=='ram_vram_finetune' or n.startswith('ram_vram_finetune.')); "
                       "print(json.dumps({'version':p.__version__,'file':p.__file__}))")
        call("installed-import", [sys.executable, "-c", import_code])
        pytest_command = [sys.executable, "-m", "pytest", source / "tests", "-c", source / "pyproject.toml",
                          "--rootdir=" + str(source), "--confcutdir=" + str(source), "-o", "addopts=",
                          "-q", "-p", "no:cacheprovider", "-p", "suite_audit", "--strict-config", "--strict-markers",
                          "--suite-audit-manifest=" + str(inventory_path),
                          "--suite-audit-installed-root=" + str(installed)]
        for collect_only in (True, False):
            label = "collect-tests" if collect_only else "cpu-tests"
            ledger_path = out / (label + "-ledger.json")
            command = pytest_command + ["--suite-audit-output=" + str(ledger_path)]
            command += ["--collect-only"] if collect_only else ["--junitxml=" + str(out / "cpu.xml"), "-o", "junit_family=xunit1"]
            result = call(label, command, required=False)
            if not ledger_path.is_file():
                raise ValueError(label + " produced no audit ledger")
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            audit = finalize_audit(inventory, ledger, result.returncode, collect_only=collect_only, installed_root=installed)
            if sha(inventory_path) != inventory_sha or ledger.get("inventory_sha256") != inventory_sha:
                audit["status"] = "FAIL"
                audit["errors"].append("inventory_identity_changed")
            if read_source_inventory(ROOT)[1] != input_inventory_sha:
                raise ValueError("original input inventory changed during run")
            if source_snapshot(source) != inventory["source_files"]:
                audit["status"] = "FAIL"
                audit["errors"].append("audited_source_changed")
            (out / (label + "-audit.json")).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
            if audit["status"] != "PASS":
                raise ValueError(label + " collection/execution audit failed")
        runtime_identity(source, installed, wheel)
        call("installed-import-after-tests", [sys.executable, "-c", import_code])
        work = out / "example"
        work.mkdir()
        cli = [sys.executable, "-m", "ram_vram_finetune.cli"]
        call("cli-help", [*cli, "--help"])
        call("console-help", [*console_entry_command(installed), "--help"])
        call("make-tiny", [*cli, "make-tiny", "--output", "tiny-model"], cwd=work)
        call("inspect-tiny", [*cli, "inspect", "--model", "tiny-model"], cwd=work)
        call("train-tiny", [*cli, "train", "--config", source / "examples/tiny.json"], cwd=work)
        call("resume-tiny", [*cli, "resume", "--config", source / "examples/resume.json", "--checkpoint", "outputs/run/checkpoint-2"], cwd=work)
        call("pip-check", [sys.executable, "-m", "pip", "check"])
        import xml.etree.ElementTree as ET
        xml = ET.parse(out / "cpu.xml")
        cases = list(xml.iter("testcase"))
        if len(cases) != len(inventory["nodeids"]) or any(list(xml.iter(k)) for k in ("failure", "error", "skipped")):
            raise ValueError("JUnit differs from exact execution audit")
        if sha(inventory_path) != inventory_sha or read_source_inventory(ROOT)[1] != input_inventory_sha:
            raise ValueError("inventory changed before final report")
        report = {"status": "PASS", "version": version, "cpu_tests": len(cases),
                  "discovery_audit": "PASS", "execution_audit": "PASS", "runtime_entries_exact": len(identity),
                  "runtime_identity": identity, "inventory_sha256": inventory_sha,
                  "original_input_inventory_sha256": input_inventory_sha,
                  "original_and_staged_source_inventory_verified": True,
                  "dependency_install": "external provisioning; verified clean CPU venv" if args.require_clean_cpu else "existing dependencies; no fresh-install claim",
                  "clean_cpu_environment_required": args.require_clean_cpu,
                  "dependency_environment": dependency_environment,
                  "gpu": "NOT_RUN", "real_model_training": "NOT_RUN", "remote_ci": "NOT_RUN",
                  "wall_seconds": time.monotonic() - started,
                  "wheel": {"path": str(wheel), "bytes": wheel.stat().st_size, "sha256": sha(wheel)},
                  "sdist": {"path": str(sdist), "bytes": sdist.stat().st_size, "sha256": sha(sdist)},
                  "source_root": str(ROOT), "output": str(out), "commands": commands,
                  "limitations": ["Dependency provisioning provenance is separately recorded; this runner builds/installs only the own project",
                                  "reviewed code/plugin trust boundary; does not resist malicious falsified pytest hooks"]}
    except Exception as exc:
        report = {"status": "FAIL", "reason": str(exc), "commands": commands,
                  "wall_seconds": time.monotonic() - started, "gpu": "NOT_RUN"}
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        raise
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("commands", "runtime_identity")}, indent=2))


if __name__ == "__main__":
    main()
