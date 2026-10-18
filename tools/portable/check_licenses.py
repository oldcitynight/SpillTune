"""Check public source attribution and exact retained license materials."""
from __future__ import annotations
import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import tarfile
import tomllib
import zipfile

GPL_SHA256='3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986'
APACHE_SHA256='77fd4710def9ec3c0f6225800e0235f15a425abd4a8b03559127fcd782612049'
REQUIRED=['COPYING','LICENSE.md','NOTICE','SOURCE_ATTRIBUTION.md','THIRD_PARTY_NOTICES.md',
          'LICENSES/transformers-APACHE-2.0.txt','legal/license-selection.json',
          'legal/source-provenance.json','legal/upstream-references.json']
def digest(data): return hashlib.sha256(data).hexdigest()
def check(root,*,wheel=None,sdist=None):
    root=Path(root); findings=[]
    try:
        for n in REQUIRED:
            if not (root/n).is_file() or (root/n).is_symlink(): findings.append('missing_or_symlink_material:'+n)
        if findings: raise ValueError('required materials missing')
        if digest((root/'COPYING').read_bytes())!=GPL_SHA256: findings.append('gpl_official_text_changed')
        if digest((root/'LICENSES/transformers-APACHE-2.0.txt').read_bytes())!=APACHE_SHA256: findings.append('retained_apache_text_changed')
        project=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))
        config=project['tool']['rvft']['licensing']
        sel=json.loads((root/'legal/license-selection.json').read_text(encoding='utf-8'))
        if config['selected-license']!='GPL-3.0-or-later' or sel['selected_license']!='GPL-3.0-or-later': findings.append('license_selection_mismatch')
        if config['scope-document']!='LICENSE.md' or config['file-review']!='legal/source-provenance.json': findings.append('incorrect_scope')
        if config['additional-link-exception'] is not False or sel['additional_link_exception'] is not False: findings.append('unsupported_link_exception')
        if sel['asset_scope']['automatic_model_data_lora_license'] is not False: findings.append('automatic_output_license_claim')
        if 'license' in project['project']: findings.append('blanket_distribution_license_metadata')
        rows=json.loads((root/'legal/source-provenance.json').read_text(encoding='utf-8'))['files']
        expected={p.relative_to(root).as_posix():digest(p.read_bytes()) for base in ['src','tests','tools/portable','launchers'] for p in (root/base).rglob('*') if p.suffix in ['.py','.c']}
        actual={r['file']:r['current_sha256'] for r in rows}
        if len(rows)!=len(actual) or expected!=actual: findings.append('current_code_provenance_incomplete_or_stale')
        for row in rows:
            if row.get('origin_kind')=='direct_alltheais_adaptation':
                if not (root/row['file']).read_text(encoding='utf-8').startswith('# Adapted from AllTheAIs '+row['upstream']['path']): findings.append('extraction_header_lost:'+row['file'])
        refs=json.loads((root/'legal/upstream-references.json').read_text(encoding='utf-8'))
        notice=(root/'NOTICE').read_text(encoding='utf-8')
        if len(refs['references'])!=30: findings.append('reference_inventory_incomplete')
        for row in refs['references']:
            if row['id'] not in notice or row['observed_sha256'] not in notice: findings.append('reference_notice_missing:'+row['id'])
            for line in row['copyright_lines']:
                if line.removeprefix('# ').strip() not in notice: findings.append('reference_copyright_missing:'+row['id'])
        if wheel:
            with zipfile.ZipFile(wheel) as z:
                names=z.namelist(); meta=next(n for n in names if n.endswith('.dist-info/METADATA')); prefix=meta.rsplit('/',1)[0]+'/licenses/'
                metadata=Parser().parsestr(z.read(meta).decode('utf-8'))
                if metadata.get('Version')!=project['project']['version']: findings.append('wheel_version_mismatch')
                if metadata.get('License-Expression') or metadata.get('License'): findings.append('wheel_blanket_license_metadata')
                for n in REQUIRED:
                    if prefix+n not in names or z.read(prefix+n)!=(root/n).read_bytes(): findings.append('wheel_material_missing_or_changed:'+n)
                if any(Path(n).suffix in ['.so','.dll','.dylib','.safetensors','.gguf','.pt','.pth'] for n in names): findings.append('unexpected_dependency_or_model_binary')
        if sdist:
            with tarfile.open(sdist) as t:
                members={'/'.join(Path(m.name).parts[1:]):m for m in t.getmembers() if m.isfile()}
                for n in REQUIRED:
                    if n not in members or t.extractfile(members[n]).read()!=(root/n).read_bytes(): findings.append('sdist_material_missing_or_changed:'+n)
        return {'status':'FAIL' if findings else 'PASS','materials_check':True,'selected_license':sel['selected_license'],'code_files_bound':len(rows),'findings':findings,'note':'Integrity and attribution checks do not grant rights to external runtimes or assets.'}
    except (KeyError,ValueError,OSError,StopIteration) as e:
        return {'status':'FAIL','materials_check':True,'findings':findings+['invalid_material_structure:'+str(e)]}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--wheel',type=Path);p.add_argument('--sdist',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    r=check(a.root,wheel=a.wheel,sdist=a.sdist)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='PASS' else 1)
