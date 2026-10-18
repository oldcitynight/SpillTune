#!/usr/bin/env python3
"""Fetch exact official CPU wheels, verify every hash, or install fully offline.

No package discovery, resolver network access, automatic retries, URL alternatives,
credentials, model downloads, source distributions or CUDA wheels are supported.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from email.parser import BytesParser

HOSTS = {'download.pytorch.org', 'pypi.org', 'files.pythonhosted.org'}
MAX_BYTES = 1073741824
MAX_SECONDS = 600

def fail(message):
    raise RuntimeError(message)

def approved(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != 'https' or p.hostname not in HOSTS or p.port not in (None, 443) or p.username or p.password or p.fragment or p.query:
        fail('Unapproved source URL')
    if p.hostname == 'download.pytorch.org' and not p.path.startswith('/whl/cpu/'):
        fail('Torch source is outside the CPU index')
    return p

class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        approved(newurl)
        # Even an otherwise approved cross-host redirect requires a new plan.
        if urllib.parse.urlsplit(req.full_url).hostname != urllib.parse.urlsplit(newurl).hostname:
            fail('Cross-host redirect rejected')
        if newurl != req.full_url:
            fail('Redirect changes the exact manifest URL')
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, data):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding="utf-8", newline="\n")
    temp.replace(path)

def clean_environment():
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('PIP_') and k not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env

def boundary():
    if platform.python_implementation() != 'CPython' or sys.version_info[:2] != (3, 14) or sys.platform != 'linux' or platform.machine() != 'x86_64':
        fail('Lock covers Linux x86_64 / CPython 3.14 only')
    import sysconfig
    if sysconfig.get_config_var('Py_GIL_DISABLED'):
        fail('Free-threaded Python is outside this lock')
    libc, version = platform.libc_ver()
    if libc != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 28):
        fail('glibc >= 2.28 is required')

def load_plan(path, lock):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get('format') != 'rvft.cpu-wheel-lock.v1':
        fail('Unexpected lock manifest format')
    entries = data['packages']
    names, files = set(), set()
    for p in entries:
        name, filename = p['name'], p['filename']
        if not isinstance(name, str) or re.sub(r'[-_.]+', '-', name).lower() != name:
            fail('Package names must be canonical lowercase identities')
        if name.startswith(('nvidia', 'cuda', 'triton')) or name in names:
            fail('GPU or duplicate package in CPU lock')
        if filename != Path(filename).name or not filename.endswith('.whl') or filename in files:
            fail('Only unique wheel basenames are allowed')
        if not re.fullmatch('[0-9a-f]{64}', p['sha256']) or type(p['bytes']) is not int or p['bytes'] <= 0:
            fail('Invalid wheel hash or size')
        u = approved(p['url'])
        if urllib.parse.unquote(u.path.rsplit('/', 1)[-1]) != filename:
            fail('URL filename differs from lock')
        if name == 'torch' and (p['version'] != '2.13.0+cpu' or u.hostname != 'download.pytorch.org' or '+cpu-' not in filename):
            fail('Torch must be the exact official CPU wheel')
        names.add(name)
        files.add(filename)
    if sum(p['bytes'] for p in entries) != data['total_wheel_bytes']:
        fail('Manifest size sum mismatch')
    expected = {f"{p['name']}=={p['version']} --hash=sha256:{p['sha256']}" for p in entries}
    actual = [line.strip() for line in lock.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if len(actual) != len(expected) or set(actual) != expected:
        fail('Requirements lock does not exactly match manifest')
    return data

def verify(plan, wheelhouse, receipt):
    expected = {p['filename'] for p in plan['packages']}
    actual = {p.name for p in wheelhouse.iterdir() if p.name.endswith('.whl')}
    if actual != expected:
        fail('Wheelhouse must contain exactly the locked wheels')
    records = []
    for p in plan['packages']:
        path = wheelhouse / p['filename']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != p['bytes'] or digest(path) != p['sha256']:
            fail('Wheel SHA256/size mismatch: ' + p['filename'])
        with zipfile.ZipFile(path) as z:
            members = z.namelist()
            if len(members) != len(set(members)) or any(n.startswith('/') or '..' in Path(n).parts for n in members):
                fail('Unsafe or duplicate wheel member')
            metadata = [n for n in members if n.count('/') == 1 and n.endswith('.dist-info/METADATA')]
            vendored_metadata = [{'member': n, 'sha256': hashlib.sha256(z.read(n)).hexdigest()}
                for n in members if n.count('/') > 1 and n.endswith('.dist-info/METADATA')]
            if len(metadata) != 1:
                fail('Wheel must contain one METADATA')
            info = z.getinfo(metadata[0])
            if info.file_size > 2000000:
                fail('METADATA exceeds limit')
            raw = z.read(metadata[0])
            msg = BytesParser().parsebytes(raw)
            norm = lambda s: re.sub(r'[-_.]+', '-', s).lower()
            if norm(msg['Name']) != norm(p['name']) or msg['Version'] != p['version']:
                fail('Wheel identity differs from lock')
            # Manifest metadata must exactly agree with the selected SHA-bound wheel.
            if sorted(msg.get_all('Requires-Dist', [])) != sorted(p['requires_dist']):
                fail('Actual wheel dependency metadata differs from plan: ' + p['name'])
            licenses = []
            for n in members:
                leaf = Path(n).name
                if '.dist-info/' not in n or not (('/licenses/' in n) or re.search(r'(?i)(license|copying|notice)', leaf)):
                    continue
                member = z.getinfo(n)
                if member.is_dir():
                    continue
                if member.file_size > 4000000:
                    fail('License member exceeds inspection limit')
                content = z.read(n)
                licenses.append({'member': n, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
            records.append({'name': p['name'], 'version': p['version'], 'filename': p['filename'], 'bytes': p['bytes'], 'sha256': p['sha256'], 'metadata_sha256': hashlib.sha256(raw).hexdigest(), 'vendored_metadata': vendored_metadata, 'license_expression': msg.get('License-Expression'), 'legacy_license': msg.get('License'), 'declared_license_files': msg.get_all('License-File', []), 'license_members': licenses})
    result = {'status': 'ALL_EXACT_WHEELS_SHA256_AND_METADATA_VERIFIED', 'platform': plan['platform'], 'packages': records, 'rights_scope': 'Exact wheel metadata and license member hashes are evidence, not distribution or combination clearance. External runtime/assets retain their actual terms.'}
    write_json(receipt, result)
    return result

def fetch(plan, wheelhouse, ledger_path, prior_bytes, prior_seconds):
    if ledger_path.exists():
        fail('Ledger already exists: no silent retry or second download run')
    if prior_bytes < 0 or prior_seconds < 0 or prior_bytes + plan['total_wheel_bytes'] > MAX_BYTES or prior_seconds >= MAX_SECONDS:
        fail('Plan exceeds remaining network budget')
    wheelhouse.mkdir(parents=True, exist_ok=True)
    ledger = {'status': 'RUNNING', 'prior_metadata_bytes': prior_bytes, 'prior_metadata_seconds': prior_seconds, 'download_bytes': 0, 'network_seconds': 0.0, 'files': [], 'redirect_policy': 'Same approved host only; exact manifest URLs; no alternatives/retries'}
    # Persist before any request; interruption must leave an explicit attempt.
    write_json(ledger_path, ledger)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirect())
    def budget():
        if prior_bytes + ledger['download_bytes'] > MAX_BYTES or prior_seconds + ledger['network_seconds'] >= MAX_SECONDS:
            fail('Network budget exhausted')
    try:
        for p in plan['packages']:
            target = wheelhouse / p['filename']
            if target.exists():
                if target.is_symlink() or target.stat().st_size != p['bytes'] or digest(target) != p['sha256']:
                    fail('Existing wheel does not match lock')
                ledger['files'].append({'filename': p['filename'], 'status': 'EXISTING_HASH_VERIFIED', 'network_bytes': 0})
                continue
            partial = target.with_name(target.name + '.part')
            if partial.exists():
                fail('Existing partial file requires explicit review')
            approved(p['url']); budget()
            rec = {'filename': p['filename'], 'url': p['url'], 'network_bytes': 0, 'status': 'STARTED'}
            ledger['files'].append(rec)
            write_json(ledger_path, ledger)
            started = time.monotonic()
            previous = ledger['network_seconds']
            try:
                req = urllib.request.Request(p['url'], method='GET', headers={'User-Agent': 'rvft-exact-cpu-wheels/1'})
                with opener.open(req, timeout=min(15, MAX_SECONDS - prior_seconds - previous)) as response:
                    approved(response.url)
                    if response.url != p['url']:
                        fail('Final URL differs from exact manifest URL')
                    if int(response.headers.get('Content-Length', '-1')) != p['bytes']:
                        fail('Content-Length differs from approved plan')
                    h = hashlib.sha256()
                    with partial.open('xb') as out:
                        while True:
                            chunk = response.read(min(1048576, p['bytes'] - rec['network_bytes'] + 1))
                            if not chunk:
                                break
                            rec['network_bytes'] += len(chunk)
                            ledger['download_bytes'] += len(chunk)
                            ledger['network_seconds'] = previous + time.monotonic() - started
                            budget()
                            if rec['network_bytes'] > p['bytes']:
                                fail('Actual wheel exceeds planned size')
                            out.write(chunk); h.update(chunk)
                    if rec['network_bytes'] != p['bytes'] or h.hexdigest() != p['sha256']:
                        fail('Downloaded wheel hash/size mismatch')
                partial.replace(target); rec['status'] = 'SHA256_VERIFIED'
            finally:
                ledger['network_seconds'] = previous + time.monotonic() - started
                write_json(ledger_path, ledger)
            budget()
        ledger['status'] = 'DOWNLOADED_ALL_EXACT_WHEELS'
    except BaseException as exc:
        ledger.update(status='FAILED_PARTIAL_PRESERVED', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(ledger_path, ledger)
    return ledger

def install(plan, wheelhouse, lock, python, receipt):
    verify(plan, wheelhouse, receipt.with_name('wheel-verification-before-install.json'))
    env = clean_environment()
    check = "import json,sys,pathlib; p=pathlib.Path(sys.prefix)/'pyvenv.cfg'; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'version':list(sys.version_info[:2]),'config':p.read_text() if p.exists() else ''}))"
    completed = subprocess.run([str(python), '-I', '-c', check], check=True, text=True, capture_output=True, env=env)
    v = json.loads(completed.stdout)
    if v['prefix'] == v['base_prefix'] or v['version'] != [3, 14] or not re.search(r'(?im)^include-system-site-packages\s*=\s*false\s*$', v['config']):
        fail('Install requires an isolated Python3.14 venv without system site packages')
    commands = [[str(python), '-I', '-m', 'pip', '--disable-pip-version-check', '--no-input', 'install', '--no-index', '--find-links', str(wheelhouse.resolve()), '--only-binary=:all:', '--require-hashes', '--no-cache-dir', '--no-compile', '-r', str(lock.resolve())], [str(python), '-I', '-m', 'pip', '--disable-pip-version-check', 'check']]
    record = {'status': 'STARTED', 'venv': v, 'commands': commands, 'steps': []}
    try:
        for command in commands:
            done = subprocess.run(command, text=True, capture_output=True, env=env)
            record['steps'].append({'returncode': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr})
            if done.returncode:
                fail('Offline pip step failed; see receipt')
        record['status'] = 'OFFLINE_HASH_LOCK_INSTALL_AND_PIP_CHECK_PASSED'
    except BaseException as exc:
        record.update(status='FAILED', error=str(exc))
        raise
    finally:
        write_json(receipt, record)
    return record

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['fetch', 'verify', 'install'])
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--wheelhouse', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--prior-metadata-bytes', type=int, default=0)
    p.add_argument('--prior-metadata-seconds', type=float, default=0)
    p.add_argument('--python', type=Path)
    a = p.parse_args()
    boundary(); plan = load_plan(a.manifest, a.lock)
    if a.action == 'fetch':
        result = fetch(plan, a.wheelhouse, a.receipt, a.prior_metadata_bytes, a.prior_metadata_seconds)
    elif a.action == 'verify':
        result = verify(plan, a.wheelhouse, a.receipt)
    else:
        if a.python is None:
            fail('--python requires the target isolated venv interpreter')
        result = install(plan, a.wheelhouse, a.lock, a.python, a.receipt)
    print(json.dumps({'status': result['status']}))

if __name__ == '__main__':
    main()
