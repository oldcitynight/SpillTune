"""Reject a plausible own wheel that additionally smuggles external payload."""
from pathlib import Path
import sys
import tomllib
import zipfile
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/portable'))
from build_deb import read_own_wheel,build
@pytest.fixture
def own_wheel(tmp_path):
    project=tomllib.loads((ROOT/'pyproject.toml').read_text())['project'];dist='spilltune-0.0.1.dist-info'
    payload={p.relative_to(ROOT/'src').as_posix():p.read_bytes() for p in (ROOT/'src/ram_vram_finetune').rglob('*.py')}
    for pattern in project['license-files']:
        for p in ROOT.glob(pattern):
            if p.is_file():payload[dist+'/licenses/'+p.relative_to(ROOT).as_posix()]=p.read_bytes()
    payload.update({dist+'/METADATA':b'Name: spilltune\nVersion: 0.0.1\n',dist+'/WHEEL':b'',dist+'/RECORD':b'',dist+'/entry_points.txt':b'',dist+'/top_level.txt':b'ram_vram_finetune\n'})
    wheel=tmp_path/'own.whl'
    with zipfile.ZipFile(wheel,'w') as z:
        for n,data in payload.items():z.writestr(n,data)
    return wheel,payload
def test_exact_own_member_set_is_accepted(own_wheel):
    wheel,payload=own_wheel
    assert read_own_wheel(ROOT,wheel)==payload
@pytest.mark.parametrize('extra',['torch/__init__.py','unexpected.py','spilltune-0.0.1.dist-info/licenses/unreviewed.txt'])
def test_extra_payload_rejected_before_output_or_compiler(own_wheel,tmp_path,extra):
    wheel,_=own_wheel
    with zipfile.ZipFile(wheel,'a') as z:z.writestr(extra,b'unexpected payload')
    output=tmp_path/'must-not-exist'
    with pytest.raises(ValueError,match='allowlist'):
        build(ROOT,wheel,output,'Example <maintainer@example.org>','/missing/compiler')
    assert not output.exists()
