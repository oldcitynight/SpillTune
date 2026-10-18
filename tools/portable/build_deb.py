"""Build the own Linux CLI deb using existing tools and an external Python runtime.

No runtime or model download/install occurs. This script only creates new output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile

def read_own_wheel(source,wheel):
    """Reject extra payload before creating an output directory or running tools."""
    source=Path(source)
    project=tomllib.loads((source/'pyproject.toml').read_text(encoding='utf-8'))['project']
    version=project['version'];dist=f'spilltune-{version}.dist-info'
    runtime={p.relative_to(source/'src').as_posix():p.read_bytes() for p in (source/'src/ram_vram_finetune').rglob('*.py')}
    materials={p.relative_to(source).as_posix():p.read_bytes() for pattern in project['license-files'] for p in source.glob(pattern) if p.is_file()}
    metadata={dist+'/'+n for n in ['METADATA','WHEEL','RECORD','entry_points.txt','top_level.txt']}
    expected=set(runtime)|metadata|{dist+'/licenses/'+n for n in materials}
    with zipfile.ZipFile(wheel) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=expected:raise ValueError('wheel member allowlist mismatch: own payload only')
        data={n:z.read(n) for n in names}
    from email.parser import Parser
    info=Parser().parsestr(data[dist+'/METADATA'].decode('utf-8'))
    if info['Version']!=version or info['Name']!=project['name']:raise ValueError('wheel metadata/version mismatch')
    if any(data[n]!=v for n,v in runtime.items()):raise ValueError('wheel/source runtime differs')
    if any(data[dist+'/licenses/'+n]!=v for n,v in materials.items()):raise ValueError('wheel/source license material differs')
    return data

def build(source,wheel,output,maintainer,cc):
    source=Path(source).resolve();wheel=Path(wheel).resolve();output=Path(output).resolve()
    if sys.platform!='linux':raise ValueError('Linux build required; native Windows/MSI is unvalidated')
    if not re.fullmatch(r'[^\r\n<>]+ <[^\s<>@]+@[^\s<>@]+>',maintainer):raise ValueError('explicit one-line Maintainer name and email required')
    project=tomllib.loads((source/'pyproject.toml').read_text(encoding='utf-8'))['project']
    version=project['version']
    if not re.fullmatch(r'\d+\.\d+\.\d+',version):raise ValueError('review Debian version mapping for non-final version')
    own_payload=read_own_wheel(source,wheel)
    output.mkdir(parents=True,exist_ok=False)
    def run(argv):return subprocess.run([str(x) for x in argv],check=True,capture_output=True,text=True,encoding='utf-8',timeout=60).stdout
    root=output/'root';(root/'DEBIAN').mkdir(parents=True);(root/'usr/bin').mkdir(parents=True)
    app=root/'usr/share/spilltune/app';app.mkdir(parents=True)
    for n,data in own_payload.items():
        target=app/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    exe=root/'usr/bin/spilltune'
    compiler=Path(shutil.which(cc) or cc).resolve(strict=True)
    run([compiler,'-std=c11','-Wall','-Wextra','-Werror','-O2','-Wl,--build-id=none','-o',exe,source/'launchers/linux/spilltune-launch.c'])
    dynamic=run(['readelf','-d',exe])
    if dynamic.count('(NEEDED)')!=1 or '[libc.so.6]' not in dynamic:raise ValueError('unexpected launcher native dependencies')
    versions=run(['readelf','--version-info',exe]);glibc=max(tuple(map(int,s.split('.'))) for s in re.findall(r'Name: GLIBC_([0-9.]+)',versions));glibc='.'.join(map(str,glibc))
    arch=run(['dpkg','--print-architecture']).strip()
    if arch!='amd64':raise ValueError('this launcher build is currently accepted only for Linux amd64')
    docs=root/'usr/share/doc/spilltune';docs.mkdir(parents=True)
    # Preserve the reviewed source layout so installed documentation links work.
    # This is own source/materials, never dependency wheels or generated assets.
    source_docs=docs/'source';source_docs.mkdir()
    allowlist=json.loads((source/'release-allowlist.json').read_text(encoding='utf-8'))['files']
    if len(allowlist)!=len(set(allowlist)):raise ValueError('duplicate source allowlist member')
    for n in allowlist:
        rel=Path(n);item=source/rel
        if rel.is_absolute() or '..' in rel.parts or item.is_symlink() or not item.is_file() or rel.suffix in ['.safetensors','.gguf','.pt','.pth','.bin','.whl','.so','.dll','.pyc']:raise ValueError('invalid source documentation member')
        target=source_docs/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(item,target)
    shutil.copyfile(source/'COPYING',docs/'COPYING')
    (docs/'README.md').write_text('# V0.0.1 Fire up\n\nSee [the complete source guide](source/README.md) for installation, CLI, capabilities and retained attribution. The source directory preserves the exact public allowlist and relative document links. External runtimes/models are not included.\n',encoding='utf-8')
    control=f'''Package: spilltune
Version: {version}
Architecture: {arch}
Maintainer: {maintainer}
Depends: python3 (>= 3.11), libc6 (>= {glibc})
Section: science
Priority: optional
Description: RAM VRAM streaming LoRA CLI with external Python environment
 V0.0.1 Fire up. Own launcher and Python tool only. PyTorch and CUDA
 remain external user-managed dependencies. No model weights are included.
'''
    (root/'DEBIAN/control').write_text(control,encoding='utf-8',newline='\n')
    for p in root.rglob('*'):p.chmod(0o755 if p.is_dir() or p==exe else 0o644)
    deb=output/f'spilltune_{version}_{arch}.deb'
    run(['dpkg-deb','--root-owner-group','--threads-max=1','-Zgzip','-z1','--build',root,deb])
    return {'deb':str(deb),'sha256':hashlib.sha256(deb.read_bytes()).hexdigest(),'bytes':deb.stat().st_size,'version':version,'architecture':arch,'depends':f'python3 (>= 3.11), libc6 (>= {glibc})','maintainer':maintainer,'maintainer_scripts':[],'embedded_external_runtime':False}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--wheel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--maintainer',required=True);p.add_argument('--cc',default='cc');a=p.parse_args()
    try:print(json.dumps(build(a.source,a.wheel,a.output,a.maintainer,a.cc),indent=2))
    except (ValueError,OSError,subprocess.SubprocessError) as e:p.exit(2,str(e)+'\n')
