"""Observable CPU environment boundary; does not install dependencies or query GPU."""
from pathlib import Path
import importlib.metadata as metadata
import sys

REQUIRED = ('torch', 'transformers', 'safetensors', 'tokenizers', 'numpy', 'pytest', 'setuptools', 'pip')


def _within(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def collect_environment_snapshot():
    import torch
    prefix = Path(sys.prefix).resolve()
    config = prefix / 'pyvenv.cfg'
    values = {}
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition('=')
            if separator:
                values[key.strip()] = value.strip()
    distributions = {}
    for dist in metadata.distributions():
        name = dist.metadata.get('Name', '')
        if name:
            distributions[name.lower().replace('_', '-')] = {
                'version': dist.version, 'root': str(Path(dist.locate_file('')).resolve())}
    return {'prefix': str(prefix), 'base_prefix': str(Path(sys.base_prefix).resolve()),
            'python': sys.version, 'executable': str(Path(sys.executable).absolute()),
            'venv_config_present': config.is_file(),
            'include_system_site_packages': values.get('include-system-site-packages'),
            'required': list(REQUIRED), 'distributions': distributions,
            'external_distribution_roots': sorted({v['root'] for v in distributions.values()
                                                    if not _within(v['root'], prefix)}),
            'external_site_package_paths': sorted({str(Path(p).resolve()) for p in sys.path if p
                and ('site-packages' in Path(p).parts or 'dist-packages' in Path(p).parts)
                and not _within(p, prefix)}),
            'torch_file': str(Path(torch.__file__).resolve()),
            'torch_version': torch.__version__, 'torch_cuda_build': torch.version.cuda,
            'torch_cuda_backend_built': torch.backends.cuda.is_built(),
            'cuda_initialized': torch.cuda.is_initialized()}


def validate_clean_cpu(snapshot):
    """Fail before candidate execution; a caller-supplied label alone is insufficient."""
    if (snapshot['prefix'] == snapshot['base_prefix'] or not snapshot['venv_config_present']
            or snapshot['include_system_site_packages'] != 'false'):
        raise ValueError('clean_cpu: isolated venv without system site packages required')
    if snapshot['external_distribution_roots'] or snapshot['external_site_package_paths']:
        raise ValueError('clean_cpu: external dependency paths detected')
    if not _within(snapshot['torch_file'], snapshot['prefix']):
        raise ValueError('clean_cpu: Torch imported outside isolated venv')
    if snapshot['torch_cuda_build'] is not None or snapshot['torch_cuda_backend_built'] or snapshot['cuda_initialized']:
        raise ValueError('clean_cpu: CPU-only Torch and uninitialized CUDA required')
    missing = set(REQUIRED) - set(snapshot['distributions'])
    if missing:
        raise ValueError('clean_cpu: required dependency distribution missing')
    forbidden = [n for n in snapshot['distributions'] if n.startswith('nvidia-') or n == 'triton']
    if forbidden:
        raise ValueError('clean_cpu: CUDA dependency distribution forbidden')
    return True
