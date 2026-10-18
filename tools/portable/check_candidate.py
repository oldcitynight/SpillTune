"""Check a strict local publication candidate without echoing secret matches."""
import argparse,hashlib,json,re
from pathlib import Path


def check(root):
    root=Path(root).resolve();policy=json.loads((root/'release-allowlist.json').read_text(encoding="utf-8"))
    allowed=set(policy['files']);manifest=root/'CANDIDATE_MANIFEST.json'
    if manifest.exists():allowed.add('CANDIDATE_MANIFEST.json')
    files={};findings=[]
    bad_suffixes={'.safetensors','.gguf','.pt','.pth','.bin','.whl','.zip','.gz','.log','.pem','.key','.pyc'}
    # Assemble machine-path patterns so the checker does not match itself.
    patterns=[('private_home',re.compile('/'+'home'+r'/[^/\s]+/')),('removable_mount',re.compile('/'+'run'+'/media/')),
        ('mac_home',re.compile('/'+'Users'+r'/[^/\s]+/')),('private_key',re.compile('-----BEGIN '+'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
        ('hf_token',re.compile(r'hf_[A-Za-z0-9]{30,}')),('github_token',re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}')),
        ('aws_key',re.compile(r'AKIA[A-Z0-9]{16}')),('unrelated_identity',re.compile('mpv'+'_web_'+'suite',re.I))]
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root).as_posix()
        if p.is_symlink():findings.append({'path':rel,'rule':'symlink'});continue
        if not p.is_file():continue
        if rel not in allowed:findings.append({'path':rel,'rule':'not_allowlisted'})
        if any(part in {'.pytest_cache','__pycache__','.git'} or part.endswith('.egg-info') for part in p.parts):findings.append({'path':rel,'rule':'generated_or_history_directory'})
        data=p.read_bytes();files[rel]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        if len(data)>2*1024**2:findings.append({'path':rel,'rule':'size_limit'})
        if p.suffix.lower() in bad_suffixes:findings.append({'path':rel,'rule':'binary_or_private_extension'})
        try:text=data.decode('utf-8')
        except UnicodeDecodeError:findings.append({'path':rel,'rule':'non_utf8'});continue
        if '\0' in text:findings.append({'path':rel,'rule':'nul_byte'})
        for name,pattern in patterns:
            if pattern.search(text):findings.append({'path':rel,'rule':name})
    for rel in sorted(set(policy['files'])-files.keys()):findings.append({'path':rel,'rule':'missing_allowlisted'})
    if manifest.exists():
        expected=json.loads(manifest.read_text(encoding="utf-8"))['files']
        if expected!={k:v for k,v in files.items() if k!='CANDIDATE_MANIFEST.json'}:findings.append({'path':'CANDIDATE_MANIFEST.json','rule':'manifest_mismatch'})
    return {'status':'PASS' if not findings else 'FAIL','files':len(files),'bytes':sum(v['bytes'] for v in files.values()),'findings':findings,'limitations':'targeted patterns and allowlist, not legal or complete secret clearance','manifest':files}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='.');parser.add_argument('--output');args=parser.parse_args();result=check(args.root)
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n', encoding="utf-8", newline="\n")
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'},indent=2));raise SystemExit(0 if result['status']=='PASS' else 1)
