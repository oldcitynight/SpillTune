"""Coverage gates must reject lost execution even when pytest returns success."""
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools/portable"
sys.path.insert(0, str(TOOLS))
import suite_audit as audit
from ci_cpu import finalize_audit

NODE = "tests/test_probe.py::test_ok"


def inventory():
    return {"format": audit.FORMAT, "source_files": {"tests/test_probe.py": "a" * 64}, "nodeids": [NODE]}


def passing_events():
    return {NODE: [{"when": phase, "outcome": "passed", "wasxfail": False}
                   for phase in ("setup", "call", "teardown")]}


def result(**kwargs):
    return audit.audit_result(inventory(), kwargs.pop("collected", [NODE]),
                              kwargs.pop("reports", passing_events()), **kwargs)


def test_exact_complete_execution_passes_without_fixed_count():
    data = inventory()
    nodes = [NODE, "tests/test_probe.py::test_new[param]"]
    data["nodeids"] = nodes
    events = passing_events()
    events[nodes[1]] = copy.deepcopy(events[NODE])
    assert audit.audit_result(data, nodes, events)["executed_and_passed"] == 2
    assert audit.audit_result(data, nodes, events)["status"] == "PASS"


@pytest.mark.parametrize("mutation", ["zero", "duplicate", "missing_file", "bad_path", "bad_hash"])
def test_malformed_inventory_rejected(mutation):
    data = inventory()
    if mutation == "zero":
        data["nodeids"] = []
    elif mutation == "duplicate":
        data["nodeids"] *= 2
    elif mutation == "missing_file":
        data["source_files"] = {"tests/other.py": "b" * 64}
    elif mutation == "bad_path":
        data["nodeids"] = ["tests/../outside.py::test_ok"]
    else:
        data["source_files"]["tests/test_probe.py"] = "wrong"
    with pytest.raises(ValueError):
        audit.validate_inventory(data)


@pytest.mark.parametrize("nodes", [[], [NODE, NODE], [NODE, "tests/test_probe.py::test_extra"]])
def test_collection_zero_duplicate_and_extra_fail(nodes):
    assert result(collected=nodes, collect_only=True)["status"] == "FAIL"


def test_same_count_wrong_test_identity_fails():
    r = result(collected=["tests/test_probe.py::test_other"], collect_only=True)
    assert r["status"] == "FAIL" and r["missing"] == [NODE] and r["extra"]


@pytest.mark.parametrize("kwargs", [{"deselected": [NODE]}, {"collection_problems": [{"outcome": "skipped"}]},
                                    {"collection_problems": [{"outcome": "failed"}]}, {"exit_code": 2}])
def test_filter_collection_problem_and_nonzero_exit_fail(kwargs):
    assert result(collect_only=True, **kwargs)["status"] == "FAIL"


@pytest.mark.parametrize("mutation", ["missing_item", "missing_call", "duplicate_call", "skipped", "failed", "xfail", "extra_item"])
def test_incomplete_or_nonpassing_execution_fails(mutation):
    events = passing_events()
    if mutation == "missing_item":
        events = {}
    elif mutation == "missing_call":
        events[NODE].pop(1)
    elif mutation == "duplicate_call":
        events[NODE].append(dict(events[NODE][1]))
    elif mutation == "extra_item":
        events["tests/test_probe.py::test_unexpected"] = []
    elif mutation == "xfail":
        events[NODE][1]["wasxfail"] = True
    else:
        events[NODE][1]["outcome"] = mutation
    assert result(reports=events)["status"] == "FAIL"


def test_parent_cannot_accept_late_nonzero_exit_or_missing_ledger():
    ledger = result()
    ledger["validation_status"] = ledger["status"]
    ledger["status"] = "RECORDED"
    assert finalize_audit(inventory(), ledger, 0, collect_only=False)["status"] == "PASS"
    assert finalize_audit(inventory(), ledger, 3, collect_only=False)["status"] == "FAIL"
    ledger["validation_status"] = "FAIL"
    assert finalize_audit(inventory(), ledger, 0, collect_only=False)["status"] == "FAIL"
    with pytest.raises(ValueError):
        finalize_audit(inventory(), {}, 0, collect_only=False)


def test_pytest_environment_filters_and_auto_plugins_are_removed(monkeypatch):
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST", "PYTHONPATH", "RVFT_QWEN35_DEVICE"):
        monkeypatch.setenv(key, "unwanted")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = audit.clean_environment()
    assert not any(k in env for k in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST", "PYTHONPATH", "RVFT_QWEN35_DEVICE"))
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"


@pytest.fixture
def tiny_suite(tmp_path):
    root = tmp_path / "fixture"
    (root / "tests").mkdir(parents=True)
    (root / "tools/portable").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.pytest.ini_options]\naddopts = "-k deliberately_no_match"\n')
    for name in ("ci_cpu.py", "suite_audit.py", "cpu_environment.py"):
        shutil.copyfile(TOOLS / name, root / "tools/portable" / name)
    (root / "tests/test_probe.py").write_text("def test_ok():\n    assert True\n")
    data = {"format": audit.FORMAT, "nodeids": [NODE], "source_files": audit.source_snapshot(root)}
    manifest = root / "tests/cpu-inventory.json"
    manifest.write_text(json.dumps(data))
    return root, manifest, data


def run_probe(tiny_suite, *, code=None, extra=(), installed_root=None):
    root, manifest, data = tiny_suite
    if code is not None:
        (root / "tests/test_probe.py").write_text(code)
        data["source_files"] = audit.source_snapshot(root)
        manifest.write_text(json.dumps(data))
    ledger = root / "ledger.json"
    env = audit.clean_environment()
    env["PYTHONPATH"] = str(root / "tools/portable")
    if installed_root is not None:
        import os
        env["PYTHONPATH"] = str(installed_root) + os.pathsep + env["PYTHONPATH"]
        extra = (*extra, "--suite-audit-installed-root=" + str(installed_root))
    extra = tuple("--ignore=" + str(root / "tests/test_probe.py") if x == "--ignore=tests/test_probe.py" else x for x in extra)
    command = [sys.executable, "-m", "pytest", str(root / "tests"), "-c", str(root / "pyproject.toml"),
               "--rootdir=" + str(root), "--confcutdir=" + str(root), "-o", "addopts=", "-q",
               "-p", "no:cacheprovider", "-p", "suite_audit", "--suite-audit-manifest=" + str(manifest),
               "--suite-audit-output=" + str(ledger), *extra]
    completed = subprocess.run(command, cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True, timeout=30)
    return completed, json.loads(ledger.read_text()) if ledger.exists() else None


def test_actual_plugin_records_events_parent_accepts_and_config_filter_is_overridden(tiny_suite):
    completed, ledger = run_probe(tiny_suite)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert ledger["status"] == "RECORDED" and ledger["pytest_version"]
    assert Counter(e["when"] for e in ledger["reports"][NODE]) == {"setup": 1, "call": 1, "teardown": 1}
    assert finalize_audit(tiny_suite[2], ledger, completed.returncode, collect_only=False)["status"] == "PASS"


@pytest.mark.parametrize("code", [
    'import pytest\n@pytest.mark.skip(reason="fixture")\ndef test_ok(): pass\n',
    'import pytest\n@pytest.mark.xfail(reason="fixture")\ndef test_ok(): assert False\n',
    'import pytest\n@pytest.mark.xfail(reason="fixture")\ndef test_ok(): pass\n',
    'import pytest\n@pytest.fixture\ndef dep(): pytest.skip("fixture")\ndef test_ok(dep): pass\n',
    'import pytest\npytest.skip("collection", allow_module_level=True)\n',
    'raise RuntimeError("collection failure")\n',
    'def helper_only(): pass\n',
    'def test_other(): pass\n',
])
def test_actual_pytest_skip_xfail_xpass_and_collection_errors_fail_closed(tiny_suite, code):
    completed, ledger = run_probe(tiny_suite, code=code)
    assert completed.returncode != 0
    assert ledger["status"] == "RECORDED" and ledger["validation_status"] == "FAIL"


@pytest.mark.parametrize("extra", [("-k", "no_such_item"), ("--deselect=" + NODE,), ("--ignore=tests/test_probe.py",)])
def test_actual_pytest_filters_cannot_pass(tiny_suite, extra):
    completed, ledger = run_probe(tiny_suite, extra=extra)
    assert completed.returncode != 0
    if ledger:
        assert ledger["validation_status"] == "FAIL"


@pytest.mark.parametrize("change", ["edit", "remove", "add"])
def test_stale_source_inventory_rejected_before_execution(tiny_suite, change):
    root, _, _ = tiny_suite
    path = root / "tests/test_probe.py"
    if change == "edit":
        path.write_text("def test_ok(): pass\n")
    elif change == "remove":
        path.unlink()
    else:
        (root / "tests/test_new.py").write_text("def test_new(): pass\n")
    completed, ledger = run_probe(tiny_suite)
    assert completed.returncode != 0 and ledger is None
    assert "differs from reviewed inventory" in completed.stderr or "empty test source tree" in completed.stderr


def test_explicit_refresh_uses_unfiltered_collection_and_requires_force(tiny_suite, monkeypatch):
    root, manifest, _ = tiny_suite
    original = manifest.read_bytes()
    env = audit.clean_environment()
    env["PYTEST_ADDOPTS"] = "-k no_such_test"
    env["PYTEST_PLUGINS"] = "nonexistent_plugin"
    command = [sys.executable, root / "tools/portable/suite_audit.py", "refresh", "--root", root,
               "--output", manifest]
    blocked = subprocess.run([str(x) for x in command], env=env, capture_output=True, text=True, timeout=30)
    assert blocked.returncode != 0 and manifest.read_bytes() == original
    refreshed = subprocess.run([str(x) for x in command] + ["--force"], env=env, capture_output=True, text=True, timeout=30)
    assert refreshed.returncode == 0, refreshed.stdout + refreshed.stderr
    assert json.loads(manifest.read_text())["nodeids"] == [NODE]


@pytest.mark.parametrize("escape", [False, True])
def test_actual_suite_attests_installed_namespace_and_rejects_prefix_lookalike(tiny_suite, escape):
    root, _, data = tiny_suite
    installed = root / "installed"
    package = installed / "ram_vram_finetune"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "fixture"\n')
    evil = root / "installed-evil/ram_vram_finetune"
    evil.mkdir(parents=True)
    (evil / "__init__.py").write_text("# fixture\n")
    code = "def test_ok(): pass\n"
    if escape:
        code = ("import sys,types\n" +
                "sys.modules['ram_vram_finetune.injected'] = types.SimpleNamespace(__file__=" +
                repr(str(evil / "__init__.py")) + ")\n" + code)
    completed, ledger = run_probe(tiny_suite, code=code, installed_root=installed)
    verdict = finalize_audit(data, ledger, completed.returncode, collect_only=False, installed_root=installed)
    assert verdict["status"] == ("FAIL" if escape else "PASS")
    assert (completed.returncode != 0) == escape
    if not escape:
        assert ledger["imported_modules"]["ram_vram_finetune"] == str(package / "__init__.py")


def test_missing_installed_root_fails_before_collection(tiny_suite):
    completed, ledger = run_probe(tiny_suite, installed_root=tiny_suite[0] / "missing-target")
    assert completed.returncode != 0 and ledger is None


@pytest.mark.parametrize("during_collection", [False, True])
def test_inventory_changes_during_run_are_rejected(tiny_suite, during_collection):
    edit = "Path(__file__).with_name('cpu-inventory.json').write_text('{}')"
    code = "from pathlib import Path\n"
    code += edit + "\ndef test_ok(): pass\n" if during_collection else "def test_ok():\n    " + edit + "\n"
    completed, ledger = run_probe(tiny_suite, code=code)
    assert completed.returncode != 0 and ledger["validation_status"] == "FAIL"
    assert "inventory_changed_during_run" in ledger["errors"]



def test_unallowlisted_new_test_is_rejected_by_actual_portable_before_copy_or_build(tiny_suite):
    root, _, _ = tiny_suite
    (root / "release-allowlist.json").write_text(json.dumps({"files": [
        "pyproject.toml", "tools/portable/ci_cpu.py", "tools/portable/suite_audit.py",
        "tools/portable/cpu_environment.py", "tests/test_probe.py", "tests/cpu-inventory.json"]}))
    (root / "tests/test_not_allowlisted.py").write_text("def test_lost(): assert False\n")
    out = root / "portable-result"
    completed = subprocess.run([sys.executable, str(root / "tools/portable/ci_cpu.py"), "--output", str(out)],
                               cwd=tempfile.gettempdir(), env=audit.clean_environment(), capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "FAIL" and "input test source differs" in report["reason"]
    assert report["commands"] == [] and not (out / "build-source").exists()
