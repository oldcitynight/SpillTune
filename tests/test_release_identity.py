"""The public CLI and package identify the same first release."""
from pathlib import Path
import sys
import tomllib
import pytest
from ram_vram_finetune import __version__
from ram_vram_finetune.cli import main
def test_package_version_matches_metadata():
    root=Path(__file__).resolve().parents[1]
    assert __version__==tomllib.loads((root/'pyproject.toml').read_text())['project']['version']=='0.0.1'
def test_cli_version_reports_release(monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['spilltune','--version'])
    with pytest.raises(SystemExit) as exc:main()
    assert exc.value.code==0
    assert capsys.readouterr().out.strip()=='spilltune 0.0.1 (Fire up)'
