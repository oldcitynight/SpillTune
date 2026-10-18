"""Clean installation guards must reject dependency borrowing and CUDA substitution."""
import copy
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/portable'))
from cpu_environment import REQUIRED, validate_clean_cpu
from suite_audit import clean_environment


def snapshot():
    return {'prefix': '/isolated/env', 'base_prefix': '/usr', 'venv_config_present': True,
            'include_system_site_packages': 'false', 'external_distribution_roots': [],
            'external_site_package_paths': [], 'torch_file': '/isolated/env/lib/torch/__init__.py',
            'torch_cuda_build': None, 'torch_cuda_backend_built': False, 'cuda_initialized': False,
            'distributions': {name: {'version': 'fixture', 'root': '/isolated/env/lib'} for name in REQUIRED}}


def test_complete_isolated_cpu_environment_is_admitted():
    assert validate_clean_cpu(snapshot()) is True


@pytest.mark.parametrize('fault', ['not_venv', 'system_sites', 'cuda_build', 'cuda_initialized',
                                  'borrowed_distribution', 'external_site', 'external_torch',
                                  'missing_required', 'cuda_distribution'])
def test_incomplete_or_borrowed_environment_fails_closed(fault):
    value = copy.deepcopy(snapshot())
    if fault == 'not_venv': value['base_prefix'] = value['prefix']
    elif fault == 'system_sites': value['include_system_site_packages'] = 'true'
    elif fault == 'cuda_build': value['torch_cuda_build'] = 'fixture'
    elif fault == 'cuda_initialized': value['cuda_initialized'] = True
    elif fault == 'borrowed_distribution': value['external_distribution_roots'] = ['/external/site-packages']
    elif fault == 'external_site': value['external_site_package_paths'] = ['/external/site-packages']
    elif fault == 'external_torch': value['torch_file'] = '/external/torch/__init__.py'
    elif fault == 'missing_required': del value['distributions']['torch']
    else: value['distributions']['nvidia-fixture'] = {'version': 'fixture', 'root': '/isolated/env/lib'}
    with pytest.raises(ValueError, match='clean_cpu:'):
        validate_clean_cpu(value)


def test_runner_removes_inherited_pip_source_and_import_overrides(monkeypatch):
    keys = ('PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'PIP_FIND_LINKS', 'PIP_CONSTRAINT',
            'PIP_REQUIREMENT', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER', 'PYTHONPATH', 'PYTHONHOME')
    for key in keys: monkeypatch.setenv(key, 'fixture-override')
    env = clean_environment()
    assert all(key not in env for key in keys)
    assert env['PIP_CONFIG_FILE'] == os.devnull and env['PIP_NO_INPUT'] == '1'
