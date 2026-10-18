"""Reviewable CPU inventory and exact pytest collection/execution audit.

CI only reads the inventory. Run the explicit ``refresh`` command after reviewing
intentional test changes; commit/review the inventory diff alongside those tests.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

FORMAT = "rvft.cpu-suite-inventory.v1"


def clean_environment():
    env = dict(os.environ)
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST",
                "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "RVFT_QWEN35_DEVICE",
                "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST", "PIP_FIND_LINKS",
                "PIP_CONSTRAINT", "PIP_REQUIREMENT", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(key, None)
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PIP_NO_CACHE_DIR="1", PIP_CONFIG_FILE=os.devnull,
               PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")
    return env


def source_snapshot(root):
    root = Path(root).resolve()
    required = [root / "pyproject.toml", root / "tools/portable/ci_cpu.py",
                root / "tools/portable/suite_audit.py"]
    files = sorted(set(required + list((root / "tools/portable").glob("*.py")))) + sorted((root / "tests").rglob("*"))
    if (root / "conftest.py").exists():
        files.append(root / "conftest.py")
    result = {}
    for path in files:
        if path.is_symlink():
            raise ValueError("symlink in audited source")
        if path.is_dir():
            continue
        if path.relative_to(root).as_posix() == "tests/cpu-inventory.json":
            continue
        if not path.is_file():
            raise ValueError("missing audited source")
        result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not any(p.startswith("tests/") for p in result):
        raise ValueError("empty test source tree")
    return result


def validate_inventory(data):
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("invalid inventory format")
    nodes = data.get("nodeids")
    files = data.get("source_files")
    if not isinstance(nodes, list) or not nodes or not isinstance(files, dict) or not files:
        raise ValueError("empty or malformed inventory")
    if any(not isinstance(n, str) or not n.startswith("tests/") or "::" not in n
           or ".." in Path(n.split("::", 1)[0]).parts for n in nodes):
        raise ValueError("invalid root-relative test identity")
    if len(set(nodes)) != len(nodes):
        raise ValueError("duplicate inventory test identity")
    for name, digest in files.items():
        if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid audited source identity")
    if any(n.split("::", 1)[0] not in files for n in nodes):
        raise ValueError("test identity absent from audited source")
    return data


def audit_result(inventory, collected, reports, *, collect_only=False, exit_code=0,
                 deselected=(), collection_problems=()):
    validate_inventory(inventory)
    expected = inventory["nodeids"]
    errors = []
    if exit_code != 0:
        errors.append("pytest_exit_nonzero")
    if not collected:
        errors.append("zero_tests_collected")
    if len(set(collected)) != len(collected):
        errors.append("duplicate_collected_identity")
    missing = sorted(set(expected) - set(collected))
    extra = sorted(set(collected) - set(expected))
    if missing:
        errors.append("missing_expected_tests")
    if extra:
        errors.append("unexpected_tests")
    if deselected:
        errors.append("tests_deselected")
    if collection_problems:
        errors.append("collection_skip_or_error")
    bad_execution = {}
    if not collect_only:
        if set(reports) - set(expected):
            errors.append("unexpected_execution_identity")
        for node in expected:
            events = reports.get(node, [])
            phases = Counter(e.get("when") for e in events)
            if (phases != Counter({"setup": 1, "call": 1, "teardown": 1})
                    or any(e.get("outcome") != "passed" or e.get("wasxfail", False)
                           for e in events)):
                bad_execution[node] = events
        if bad_execution:
            errors.append("not_all_expected_tests_executed_and_passed")
    return {"status": "FAIL" if errors else "PASS", "mode": "collection" if collect_only else "execution",
            "expected": len(expected), "collected": len(collected),
            "executed_and_passed": 0 if collect_only else len(expected) - len(bad_execution),
            "errors": errors, "missing": missing, "extra": extra,
            "deselected": list(deselected), "collection_problems": list(collection_problems),
            "bad_execution": bad_execution, "collected_nodeids": collected,
            "reports": reports, "pytest_exit_code": int(exit_code)}



def validate_import_paths(paths, installed_root):
    root = Path(installed_root).resolve() / "ram_vram_finetune"
    if not root.is_dir() or not paths or paths.get("ram_vram_finetune") != str(root / "__init__.py"):
        raise ValueError("installed namespace root missing or mismatched")
    for name, value in paths.items():
        if not (name == "ram_vram_finetune" or name.startswith("ram_vram_finetune.")):
            raise ValueError("unexpected imported namespace")
        path = Path(value).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError("imported runtime escaped installed package")
    return paths


def imported_module_paths(installed_root):
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if name == "ram_vram_finetune" or name.startswith("ram_vram_finetune."):
            file = getattr(module, "__file__", None)
            if not file:
                raise ValueError("imported runtime has no source file")
            paths[name] = str(Path(file).resolve())
    return validate_import_paths(paths, installed_root)


def pytest_addoption(parser):
    group = parser.getgroup("rvft suite audit")
    group.addoption("--suite-audit-manifest")
    group.addoption("--suite-audit-installed-root")
    group.addoption("--suite-audit-output")
    group.addoption("--suite-audit-record", help="Explicit maintenance only; never used by CI")


def pytest_configure(config):
    record = config.getoption("--suite-audit-record")
    manifest = config.getoption("--suite-audit-manifest")
    output = config.getoption("--suite-audit-output")
    installed_root = config.getoption("--suite-audit-installed-root")
    if bool(record) == bool(manifest) or (manifest and not output):
        raise pytest.UsageError("choose explicit record, or manifest plus audit output")
    if record and not config.option.collectonly:
        raise pytest.UsageError("inventory recording requires collect-only")
    root = Path(config.rootpath)
    try:
        snapshot = source_snapshot(root)
        if installed_root:
            importlib.import_module("ram_vram_finetune")
            imported_module_paths(installed_root)
        inventory_content = None if record else Path(manifest).read_bytes()
        manifest_sha = None if record else hashlib.sha256(inventory_content).hexdigest()
        inventory = None if record else validate_inventory(json.loads(inventory_content))
        if inventory and snapshot != inventory["source_files"]:
            raise ValueError("audited source differs from reviewed inventory; explicit refresh required")
    except (ValueError, OSError, ImportError) as exc:
        raise pytest.UsageError(str(exc)) from exc
    config.pluginmanager.register(SuiteAudit(root, inventory, record, output, installed_root, manifest, manifest_sha), "rvft_suite_auditor")


class SuiteAudit:
    def __init__(self, root, inventory, record, output, installed_root, manifest, manifest_sha):
        self.root, self.inventory, self.record, self.output = root, inventory, record, output
        self.installed_root = str(Path(installed_root).resolve()) if installed_root else None
        self.manifest, self.manifest_sha = manifest, manifest_sha
        self.collected, self.reports, self.deselected, self.problems = [], {}, [], []

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report):
        if report.outcome != "passed":
            self.problems.append({"nodeid": report.nodeid, "outcome": report.outcome})

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(report.nodeid, []).append(
            {"when": report.when, "outcome": report.outcome,
             "wasxfail": hasattr(report, "wasxfail")})

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        if self.record:
            if exitstatus or not self.collected or self.problems or self.deselected:
                session.exitstatus = 1
                return
            data = {"format": FORMAT, "source_files": source_snapshot(self.root),
                    "nodeids": sorted(self.collected)}
            validate_inventory(data)
            Path(self.record).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
            return
        result = audit_result(self.inventory, self.collected, self.reports,
                              collect_only=session.config.option.collectonly,
                              exit_code=exitstatus, deselected=self.deselected,
                              collection_problems=self.problems)
        # Catch source changes during collection/execution as well as before it.
        if source_snapshot(self.root) != self.inventory["source_files"]:
            result["errors"].append("audited_source_changed_during_run")
            result["status"] = "FAIL"
        result["inventory_sha256"] = self.manifest_sha
        if (not Path(self.manifest).is_file() or
                hashlib.sha256(Path(self.manifest).read_bytes()).hexdigest() != self.manifest_sha):
            result["errors"].append("inventory_changed_during_run")
            result["status"] = "FAIL"
        result["installed_root"] = self.installed_root
        result["imported_modules"] = {}
        if self.installed_root:
            try:
                result["imported_modules"] = imported_module_paths(self.installed_root)
            except ValueError as exc:
                result["status"] = "FAIL"
                result["errors"].append(str(exc))
        passed = result["status"] == "PASS"
        result["validation_status"] = result["status"]
        result["status"] = "RECORDED"
        result["pytest_version"] = pytest.__version__
        result["plugins"] = sorted(name for name, _ in session.config.pluginmanager.list_name_plugin())
        Path(self.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        if not passed:
            session.exitstatus = 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    refresh = sub.add_parser("refresh")
    refresh.add_argument("--root", default=Path(__file__).resolve().parents[2], type=Path)
    refresh.add_argument("--output", type=Path, required=True)
    refresh.add_argument("--force", action="store_true", help="replace an existing reviewed inventory")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() and not args.force:
        parser.error("inventory exists; review changes and use --force deliberately")
    env = clean_environment()
    env["PYTHONPATH"] = str(root / "tools/portable") + os.pathsep + str(root / "src")
    command = [sys.executable, "-m", "pytest", str(root / "tests"), "-c", str(root / "pyproject.toml"),
               "--rootdir=" + str(root), "--confcutdir=" + str(root), "-o", "addopts=",
               "-p", "no:cacheprovider", "-p", "suite_audit", "--collect-only", "-q",
               "--suite-audit-record=" + str(output)]
    raise SystemExit(subprocess.run(command, cwd=tempfile.gettempdir(), env=env, timeout=180).returncode)


if __name__ == "__main__":
    main()
