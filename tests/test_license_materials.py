"""Public exports retain source/notice obligations without private audit history."""
import json
from pathlib import Path
import shutil
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/portable'))
from check_licenses import check
def test_current_materials_and_origin_mapping():
    report=check(ROOT)
    assert report['status']=='PASS',report['findings']
    assert report['selected_license']=='GPL-3.0-or-later'
@pytest.fixture
def copy_candidate(tmp_path):
    for n in json.loads((ROOT/'release-allowlist.json').read_text())['files']:
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    return tmp_path
@pytest.mark.parametrize('mutation',['missing_gpl','changed_gpl','lost_apache','blanket_metadata','missing_origin','stale_origin','notice_copyright','notice_hash','reference_inventory','link_exception','automatic_output','configured_exception','configured_scope'])
def test_material_corruption_is_rejected(copy_candidate,mutation):
    root=copy_candidate
    if mutation=='missing_gpl':(root/'COPYING').unlink()
    elif mutation=='changed_gpl':(root/'COPYING').write_bytes(b'changed')
    elif mutation=='lost_apache':(root/'LICENSES/transformers-APACHE-2.0.txt').unlink()
    elif mutation in ['blanket_metadata','configured_exception','configured_scope']:
        p=root/'pyproject.toml';text=p.read_text(encoding='utf-8')
        if mutation=='blanket_metadata':text=text.replace('[project]\n','[project]\nlicense = "GPL-3.0-or-later"\n')
        elif mutation=='configured_exception':text=text.replace('additional-link-exception = false','additional-link-exception = true')
        else:text=text.replace('scope-document = "LICENSE.md"','scope-document = "missing.md"')
        p.write_text(text,encoding='utf-8')
    elif mutation in ['missing_origin','stale_origin']:
        p=root/'legal/source-provenance.json';d=json.loads(p.read_text())
        if mutation=='missing_origin':d['files'].pop()
        else:d['files'][0]['current_sha256']='0'*64
        p.write_text(json.dumps(d),encoding='utf-8')
    elif mutation.startswith('notice_'):
        p=root/'NOTICE';text=p.read_text()
        if mutation=='notice_copyright':text=text.replace('Copyright 2025 The HuggingFace Team. All rights reserved.','removed')
        else:text=text.replace('c0889ac0b0a28b6b40ce7a6ecad6c491473626147c603348595181f0adc4ab4c','0'*64)
        p.write_text(text,encoding='utf-8')
    elif mutation=='reference_inventory':
        p=root/'legal/upstream-references.json';d=json.loads(p.read_text());d['references'].pop();p.write_text(json.dumps(d),encoding='utf-8')
    else:
        p=root/'legal/license-selection.json';d=json.loads(p.read_text())
        if mutation=='link_exception':d['additional_link_exception']=True
        else:d['asset_scope']['automatic_model_data_lora_license']=True
        p.write_text(json.dumps(d),encoding='utf-8')
    assert check(root)['status']=='FAIL'
