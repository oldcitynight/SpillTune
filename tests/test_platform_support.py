"""Platform branches are simulations; only the host execution is native."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from ram_vram_finetune import platform_support as support


@pytest.mark.parametrize('platform_name,expected', [('linux', 123 * 1024), ('darwin', 123), ('win32', None), ('unknown', None)])
def test_rss_units_or_explicit_unavailable(monkeypatch, platform_name, expected):
    fake = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=123))
    monkeypatch.setattr(support, '_resource', fake)
    assert support.rss_high_water_bytes(platform_name=platform_name) == expected


def test_missing_resource_does_not_break_import_or_fabricate_memory(monkeypatch):
    import builtins
    import importlib.util
    real_import = builtins.__import__
    def without_resource(name, *args, **kwargs):
        if name == 'resource':
            raise ImportError('simulated Windows resource absence')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_resource)
    spec = importlib.util.spec_from_file_location('platform_probe', support.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.rss_high_water_bytes() is None
    assert module.platform_capabilities()['rss_high_water_source'] == 'unavailable'


def test_windows_directory_sync_explicitly_unsupported_without_open(monkeypatch):
    def forbidden(*args): raise AssertionError('must not open a Windows directory fd')
    monkeypatch.setattr(support, 'os', SimpleNamespace(open=forbidden))
    assert support.sync_directory('C:/example', platform_name='win32') is False


def test_posix_sync_failure_propagates_and_closes_descriptor(monkeypatch):
    calls = []
    def fail(fd):
        calls.append(('fsync', fd)); raise OSError('injected directory failure')
    fake = SimpleNamespace(O_RDONLY=0, open=lambda *_: 17, fsync=fail, close=lambda fd: calls.append(('close', fd)))
    monkeypatch.setattr(support, 'os', fake)
    with pytest.raises(OSError, match='injected'):
        support.sync_directory('/example', platform_name='linux')
    assert calls == [('fsync', 17), ('close', 17)]


def test_non_ascii_checkpoint_and_existing_lock_are_preserved(tmp_path):
    from ram_vram_finetune.checkpoint_io.state import _publish, _read
    target = tmp_path / '模型 with spaces' / '检查点'
    tensor = torch.tensor([1.0, 2.0])
    _publish(target, {'value': tensor}, {'说明': '可恢复'})
    tensors, metadata = _read(target)
    assert torch.equal(tensors['value'], tensor) and metadata == {'说明': '可恢复'}
    lock = target.parent / '.occupied.lock'; lock.write_text('held', encoding='utf-8')
    with pytest.raises(FileExistsError):
        _publish(target.parent / 'occupied', {'value': tensor}, {})
    assert lock.read_text(encoding='utf-8') == 'held'
    assert not (target.parent / 'occupied').exists()
    assert not list(target.parent.glob('.stage-*'))


def test_unavailable_rss_metric_propagates_through_real_cpu_training(tmp_path, monkeypatch):
    from ram_vram_finetune.fixture import make_tiny
    from ram_vram_finetune.config import TrainConfig
    from ram_vram_finetune.training import engine
    root = tmp_path / '训练 path'; make_tiny(root, layers=1, adapter='llama')
    monkeypatch.setattr(engine, 'rss_high_water_bytes', lambda: None)
    monkeypatch.setattr(engine, 'platform_capabilities', lambda: {'platform': 'simulated', 'rss_high_water_source': 'unavailable', 'directory_fsync_supported': False, 'native_windows_validation': 'NOT_ESTABLISHED'})
    report = engine.run(TrainConfig(str(root), str(root / 'tokens.jsonl'), str(tmp_path / '结果 path'), adapter='llama', steps=1))
    assert report['status'] == 'completed' and report['metrics'][0]['rss_high_water_bytes'] is None
    assert report['platform_capabilities']['rss_high_water_source'] == 'unavailable'


@pytest.mark.parametrize('platform_name,expected', [('win32', ['C:\\Program Files\\SpillTune\\Scripts\\spilltune.exe']), ('linux', None)])
def test_platform_console_command_preserves_path_arguments(platform_name, expected):
    tools = Path(__file__).resolve().parents[1] / 'tools/portable'
    sys.path.insert(0, str(tools))
    try:
        from ci_cpu import console_entry_command
        if expected is not None:
            assert console_entry_command('C:/Program Files/SpillTune', platform_name=platform_name) == expected
        else:
            assert console_entry_command('/tmp/path with spaces', platform_name=platform_name) == [sys.executable, '/tmp/path with spaces/bin/spilltune']
    finally:
        sys.path.remove(str(tools))


def test_production_text_io_is_explicit_utf8():
    root = Path(__file__).resolve().parents[1]
    for base in [root / 'src', root / 'tools/portable']:
        for path in base.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {'read_text', 'write_text'}:
                    values = {k.arg: k.value for k in node.keywords}
                    assert isinstance(values.get('encoding'), ast.Constant) and values['encoding'].value == 'utf-8', (path, node.lineno)


def test_peft_failed_validation_releases_loaded_owner_before_cleanup(tmp_path, monkeypatch):
    from ram_vram_finetune.checkpoint_io import state
    released = []
    class Loaded(dict):
        def __del__(self): released.append('loaded_owner_released')
    projection = SimpleNamespace(A=torch.ones(1, 2), B=torch.ones(2, 1))
    block = SimpleNamespace(index=0, adapters={'q': projection})
    adapter = SimpleNamespace(blocks=[block], identity='fixture', artifact_key=lambda i,n,f: f'{i}/{n}/{f}')
    config = SimpleNamespace(rank=1, alpha=2.0, model=str(tmp_path / 'base'))
    monkeypatch.setattr(state, 'load_file', lambda _: Loaded({'wrong/key': torch.ones(1)}))
    original = state.shutil.rmtree
    def cleanup(path):
        assert released == ['loaded_owner_released']
        original(path)
    monkeypatch.setattr(state.shutil, 'rmtree', cleanup)
    with pytest.raises(RuntimeError, match='PEFT tensor reload failed'):
        state.export_peft(tmp_path / 'adapter', adapter, config)
    assert not (tmp_path / 'adapter').exists()
    assert not (tmp_path / '.adapter.lock').exists()
    assert not list(tmp_path.glob('.peft-*'))
