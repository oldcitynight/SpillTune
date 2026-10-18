"""Locked provisioning rejects unexpected sources and corruption before installation."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/portable'))
import fetch_cpu_wheels as fetch


@pytest.mark.parametrize('url', ['http://pypi.org/file.whl', 'https://external.invalid/file.whl',
    'https://user:fixture@pypi.org/file.whl', 'https://download.pytorch.org/whl/cu130/file.whl',
    'https://pypi.org:444/file.whl', 'https://pypi.org/file.whl?fixture=1',
    'https://pypi.org/file.whl#fixture'])
def test_unapproved_or_credential_sources_are_rejected(url):
    with pytest.raises(RuntimeError, match='Unapproved|CPU index'):
        fetch.approved(url)


def test_redirect_is_rejected_before_a_changed_request_is_created():
    request = urllib.request.Request('https://pypi.org/one.whl')
    with pytest.raises(RuntimeError, match='redirect|Redirect'):
        fetch.Redirect().redirect_request(request, None, 302, 'fixture', {}, 'https://pypi.org/two.whl')


def test_manifest_and_lock_have_identical_complete_package_sets():
    root = Path(__file__).resolve().parents[1]
    plan = fetch.load_plan(root / 'requirements/cpu-linux-x86_64-cp314.manifest.json',
                           root / 'requirements/cpu-linux-x86_64-cp314.lock')
    assert len(plan['packages']) == 37
    assert plan['total_wheel_bytes'] == 244633870
    assert all(not p['name'].startswith(('nvidia', 'cuda', 'triton')) for p in plan['packages'])


@pytest.mark.parametrize('fault', ['gpu_name', 'noncanonical_name', 'unsafe_filename', 'boolean_size', 'lock_changed'])
def test_mutated_manifest_or_lock_fails_before_network(tmp_path, fault):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'requirements/cpu-linux-x86_64-cp314.manifest.json').read_text())
    lock = (root / 'requirements/cpu-linux-x86_64-cp314.lock').read_text()
    if fault == 'gpu_name': plan['packages'][0]['name'] = 'nvidia-fixture'
    elif fault == 'noncanonical_name': plan['packages'][0]['name'] = 'NVIDIA_Fixture'
    elif fault == 'unsafe_filename': plan['packages'][0]['filename'] = '../fixture.whl'
    elif fault == 'boolean_size': plan['packages'][0]['bytes'] = True
    else: lock += '\nfixture==1 --hash=sha256:' + '0' * 64 + '\n'
    p = tmp_path / 'manifest.json';p.write_text(json.dumps(plan))
    k = tmp_path / 'requirements.lock';k.write_text(lock)
    with pytest.raises(RuntimeError): fetch.load_plan(p, k)


def test_corrupt_wheel_is_rejected_without_installer_execution(tmp_path):
    p = tmp_path / 'fixture.whl';p.write_bytes(b'fixture')
    plan = {'platform': {}, 'packages': [{'name': 'fixture', 'version': '1', 'filename': p.name,
            'bytes': p.stat().st_size, 'sha256': '0' * 64, 'requires_dist': []}]}
    with pytest.raises(RuntimeError, match='SHA256/size'):
        fetch.verify(plan, tmp_path, tmp_path / 'receipt.json')
    assert not (tmp_path / 'receipt.json').exists()


@pytest.mark.parametrize('duplicate_root', [False, True])
def test_only_unique_top_level_metadata_identifies_distribution(tmp_path, duplicate_root):
    p = tmp_path / 'fixture-1-py3-none-any.whl'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('fixture-1.dist-info/METADATA', 'Metadata-Version: 2.4\nName: fixture\nVersion: 1\n\n')
        z.writestr('fixture/_vendor/dependency-2.dist-info/METADATA', 'Name: dependency\nVersion: 2\n\n')
        if duplicate_root:
            z.writestr('second-1.dist-info/METADATA', 'Name: second\nVersion: 1\n\n')
    plan = {'platform': {}, 'packages': [{'name': 'fixture', 'version': '1', 'filename': p.name,
        'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'requires_dist': []}]}
    if duplicate_root:
        with pytest.raises(RuntimeError, match='one METADATA'):
            fetch.verify(plan, tmp_path, tmp_path / 'receipt.json')
    else:
        result = fetch.verify(plan, tmp_path, tmp_path / 'receipt.json')
        assert result['status'] == 'ALL_EXACT_WHEELS_SHA256_AND_METADATA_VERIFIED'
        assert len(result['packages'][0]['vendored_metadata']) == 1
