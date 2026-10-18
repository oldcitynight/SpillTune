import hashlib
import signal
from types import SimpleNamespace

import pytest
import torch

from ram_vram_finetune.memory import pinned_layout
from ram_vram_finetune.checkpoint_io.state import _snapshot, _sha
from ram_vram_finetune.config import TrainConfig
from ram_vram_finetune.fixture import make_tiny
from ram_vram_finetune.training.engine import run


def test_arena_planner_metadata_alignment_and_no_tensor_references():
    blocks = [SimpleNamespace(host_weights={'large': torch.empty(96*1024, dtype=torch.uint8, device='meta')})]
    # Metadata shape distribution at 1/1024 scale. Actual CPU storage < 1 MiB.
    blocks = [SimpleNamespace(host_weights={'mlp': torch.zeros(96*1024, dtype=torch.uint8),
                                           'attn': torch.zeros(32*1024, dtype=torch.uint8)}) for _ in range(3)]
    layout = pinned_layout(blocks, arena_limit=128*1024)
    assert sum(a['capacity'] for a in layout) == 3*128*1024
    assert all(offset % 256 == 0 for a in layout for _, _, offset, _ in a['items'])
    assert all(type(v) in (int, str) for a in layout for item in a['items'] for v in item)


def test_arena_large_tensor_and_budget_rounding():
    block = SimpleNamespace(host_weights={'huge': torch.zeros(129*1024, dtype=torch.uint8)})
    layout = pinned_layout([block], arena_limit=128*1024)
    assert len(layout) == 1 and layout[0]['capacity'] == 256*1024
    with pytest.raises(ValueError, match='contiguous'):
        pinned_layout([SimpleNamespace(host_weights={'bad': torch.ones(3,4).t()})])


def test_streaming_hash_and_independent_single_cpu_snapshot(tmp_path, monkeypatch):
    file = tmp_path/'data'; data = b'x'*(3*1024**2+7); file.write_bytes(data)
    monkeypatch.setattr(type(file), 'read_bytes', lambda *_: (_ for _ in ()).throw(AssertionError('full read forbidden')))
    assert _sha(file) == hashlib.sha256(data).hexdigest()
    source = torch.arange(12.).view(3,4).t()
    saved = _snapshot({'v': source})['v']
    assert saved.is_contiguous() and torch.equal(source, saved) and saved.data_ptr() != source.data_ptr()
    source.zero_(); assert torch.count_nonzero(saved) == 11


def test_cancel_during_atomic_export_reports_cancelled_keeps_complete_artifacts(tmp_path, monkeypatch):
    import ram_vram_finetune.checkpoint_io.state as state
    original = state.export_adapter
    def export(*args, **kwargs):
        signal.raise_signal(signal.SIGTERM)
        return original(*args, **kwargs)
    monkeypatch.setattr(state, 'export_adapter', export)
    model = make_tiny(tmp_path/'model', layers=2)
    output = tmp_path/'run'
    result = run(TrainConfig(str(model), str(model/'tokens.jsonl'), str(output), steps=1))
    assert result['status'] == 'cancelled' and result['optimizer_steps'] == 1
    assert (output/'adapter/COMPLETE').exists() and (output/'peft-adapter/adapter_config.json').exists()
