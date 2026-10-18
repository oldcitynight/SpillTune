"""One-tensor-at-a-time source loader and self-contained quantized archives."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from .codecs import stride_for,Record,check_format,encode,validate_spec,encoded_size,DTYPES

MARKER='rvft_quantization.json'
SIDECARS={'config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
          'chat_template.jinja','vocab.json','merges.txt','added_tokens.json','LICENSE','LICENSE.txt','LICENSE.md','NOTICE','README.md'}


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk:=handle.read(1024**2):h.update(chunk)
    return h.hexdigest()


def hash_tensor(h,name,tensor):
    h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
    view=tensor.detach().view(torch.uint8).reshape(-1)
    for start in range(0,view.numel(),4*1024**2):h.update(memoryview(view[start:start+4*1024**2].numpy()))


def archive_entries(root, family):
    """Metadata/key/shape validation precedes any tensor payload loading."""
    root=Path(root);file=root/MARKER
    if not file.exists():return None
    meta=json.loads(file.read_text(encoding="utf-8"))
    if set(meta)!= {'format','family','quantization','source_identity','storage_identity','entries','files'} or meta['format']!='rvft.quantized_base.v1' or meta['family']!=family:
        raise ValueError('unsupported_quantization: archive schema/family')
    check_format(meta['quantization'])
    if not isinstance(meta['entries'],dict) or not meta['entries']:raise ValueError('empty quantized archive')
    if (root/'COMPLETE').read_text(encoding="utf-8")!=file_hash(file)+'\n':raise ValueError('quantized archive manifest mismatch')
    files=meta['files']
    if 'config.json' not in files:raise ValueError('missing bound model configuration')
    for name,digest in files.items():
        if Path(name).name!=name or (name not in SIDECARS and not (name.startswith('tensor-') and name.endswith('.safetensors'))) or not (root/name).is_file() or file_hash(root/name)!=digest:
            raise ValueError('quantized archive file/hash mismatch')
    entries={};used=set()
    for key,entry in meta['entries'].items():
        if set(entry)!= {'file','shape','dtype','spec','stored_bytes'} or entry['file'] in used or entry['file'] not in files:
            raise ValueError('duplicate/missing quantized tensor file')
        used.add(entry['file']);spec=entry['spec']
        if not isinstance(entry['shape'],list) or not entry['shape'] or any(type(n)is not int or n<1 for n in entry['shape']) or entry['dtype'] not in {'F32','BF16'}:
            raise ValueError('invalid logical tensor schema')
        if spec.get('shape')!=entry['shape'] or spec.get('codec') not in {'raw',meta['quantization']}:
            raise ValueError('quantized logical shape/codec mismatch')
        selected=spec.get('codec')!='raw';count=math.prod(entry['shape'])
        if spec.get('version')!=1 or spec.get('dtype') not in DTYPES:
            raise ValueError('invalid quantized archive policy/version/dtype')
        expected_fields={'version','shape','dtype','codec'}|({'group_size','numel'} if selected else set())
        if set(spec)!=expected_fields or (selected and (spec.get('group_size')!=128 or spec.get('numel')!=count)):
            raise ValueError('invalid quantized archive group/numel')
        expected_bytes=encoded_size(entry['shape'],entry['dtype'],meta['quantization'],selected=selected,raw_dtype=DTYPES[spec['dtype']])
        if type(entry['stored_bytes'])is not int or entry['stored_bytes']!=expected_bytes:raise ValueError('invalid stored byte budget')
        with safe_open(str(root/entry['file']),framework='pt',device='cpu') as handle:
            headers={k:(handle.get_slice(k).get_shape(),handle.get_slice(k).get_dtype()) for k in handle.keys()}
        expected={'value':(entry['shape'],'BF16' if spec.get('dtype')=='bfloat16' else 'F32')} if spec.get('codec')=='raw' else {
            'payload':([((count+127)//128)*(stride_for(spec['codec']))],'U8'),
            'scales':([(count+127)//128],'F32')}
        if headers!=expected:raise ValueError('quantized archive tensor headers mismatch')
        entries[key]={**entry,'file':str(root/entry['file'])}
    return entries


def scan_entries(root,family):
    archive=archive_entries(root,family)
    if archive is not None:return archive
    entries={}
    for shard in sorted(Path(root).glob('*.safetensors')):
        with safe_open(str(shard),framework='pt',device='cpu') as handle:
            for key in handle.keys():
                if key in entries:raise ValueError('duplicate checkpoint tensor')
                value=handle.get_slice(key)
                entries[key]={'file':str(shard),'shape':value.get_shape(),'dtype':value.get_dtype()}
    if not entries and list(Path(root).glob('*.gguf')):raise ValueError('unsupported_quantization: GGUF input is not RVFT storage')
    return entries


class WeightLoader:
    def __init__(self,path,report,mode,*,pin_keys=(),host_budget_bytes=None,pinned_arena_limit=128*1024**2):
        self.path=Path(path);self.report=report;self.mode=mode
        if mode!='none':check_format(mode)
        self.archive=json.loads((self.path/MARKER).read_text(encoding="utf-8")) if (self.path/MARKER).exists() else None
        if self.archive and mode!=self.archive['quantization']:raise ValueError('unsupported_quantization: explicitly select archive format')
        self.original=hashlib.sha256();self.original.update(json.dumps(report['text_config'],sort_keys=True).encode())
        self.stored=hashlib.sha256();self.stored.update(json.dumps({'format':mode,'version':1,
            'compute':'bf16' if mode!='none' else 'source','config':report['text_config']},sort_keys=True).encode())
        self.read_keys=[];self.stored_bytes=0;self.max_source_tensor_bytes=0
        self.quantized_tensors=0
        self.pinned_parts={};self.pinned_owners={};self.pinned_capacities=[]
        self.mapped_source=bool(pin_keys)
        if pin_keys:
            if type(pinned_arena_limit) is not int or pinned_arena_limit<1 or pinned_arena_limit & (pinned_arena_limit-1):
                raise ValueError('pinned arena limit must be a positive power of two')
            from ..memory import pinned_layout_items
            items=[]
            for key in pin_keys:
                entry=report['entries'][key];count=math.prod(entry['shape'])
                if mode!='none' and key in report['quantization_matrix_keys']:
                    groups=(count+127)//128
                    parts={'payload':groups*stride_for(mode),'scales':groups*4}
                else:parts={'value':estimated_bytes(report,mode,key)}
                items.extend((size,key,part) for part,size in parts.items())
            layout=pinned_layout_items(items,pinned_arena_limit)
            self.pinned_capacities=[a['capacity'] for a in layout]
            if host_budget_bytes is None or sum(self.pinned_capacities)>host_budget_bytes:
                raise MemoryError('resource_budget: direct pinned loading exceeds host allowance')
            for index,arena in enumerate(layout):
                for key,part,offset,size in arena['items']:self.pinned_parts[key,part]=(index,offset,size)

    def read(self,key):
        entry=self.report['entries'][key]
        selected=key in self.report['quantization_matrix_keys']
        raw_dtype=torch.float32 if key in self.report['quantization_fp32_keys'] else torch.bfloat16
        if self.archive:
            expected_codec=self.mode if selected else 'raw'
            expected_dtype='bfloat16' if selected else str(raw_dtype).split('.')[-1]
            if entry['spec']['codec']!=expected_codec or entry['spec']['dtype']!=expected_dtype:
                raise ValueError('quantized archive violates adapter-declared storage policy')
            record=Record(entry['spec'],load_file(entry['file']))
            validate_spec(record.spec,record.parts)
        else:
            with safe_open(entry['file'],framework='pt',device='cpu') as handle:
                value=handle.get_tensor(key).detach().contiguous()
                if not self.mapped_source:value=value.clone()
            self.max_source_tensor_bytes=max(self.max_source_tensor_bytes,value.numel()*value.element_size())
            hash_tensor(self.original,key,value)
            record=encode(value,self.mode,quantize=selected,raw_dtype=raw_dtype)
            del value
        for part,tensor in list(record.parts.items()):
            planned=self.pinned_parts.get((key,part))
            if planned is not None:
                index,offset,size=planned
                if index not in self.pinned_owners:
                    self.pinned_owners[index]=torch.empty(self.pinned_capacities[index],dtype=torch.uint8,pin_memory=True)
                view=self.pinned_owners[index].narrow(0,offset,size).view(tensor.dtype).view(tensor.shape)
                view.copy_(tensor);record.parts[part]=view
        self.stored.update(key.encode());self.stored.update(json.dumps(record.spec,sort_keys=True).encode())
        if record.spec['codec']!='raw':self.quantized_tensors+=1
        for part,tensor in sorted(record.parts.items()):
            hash_tensor(self.stored,part,tensor);self.stored_bytes+=tensor.numel()*tensor.element_size()
        self.read_keys.append(key)
        return record

    def finish(self):
        if self.read_keys!=self.report['required']:raise ValueError('source loading order/inventory mismatch')
        if self.mode!='none' and self.quantized_tensors<1:raise ValueError('unsupported_quantization: no eligible decoder matrices; no silent BF16 fallback')
        source=self.archive['source_identity'] if self.archive else self.original.hexdigest()
        storage=self.stored.hexdigest()
        if self.archive and storage!=self.archive['storage_identity']:raise ValueError('quantized base identity mismatch')
        identity=source if self.mode=='none' else hashlib.sha256((source+storage).encode()).hexdigest()
        self.source_identity,self.storage_identity=source,storage
        return identity


def estimated_bytes(report,mode,key):
    entry=report['entries'][key]
    return entry.get('stored_bytes',encoded_size(entry['shape'],entry['dtype'],mode,
        selected=key in report['quantization_matrix_keys'],raw_dtype=torch.float32 if key in report['quantization_fp32_keys'] else torch.bfloat16))


def require_export_capability(report, mode):
    """Consume explicit adapter policy; absent or malformed declarations deny."""
    check_format(mode)
    capability=report.get('quantized_export')
    if not isinstance(capability,dict) or set(capability)!={'version','formats','rejection_reason'}:
        raise ValueError('unsupported_quantization: missing or malformed adapter export capability')
    formats=capability['formats']
    if type(capability['version']) is not int or capability['version']!=1 or not isinstance(formats,list) or any(not isinstance(f,str) for f in formats) or len(set(formats))!=len(formats) or not isinstance(capability['rejection_reason'],str):
        raise ValueError('unsupported_quantization: malformed adapter export capability')
    for declared in formats:check_format(declared)
    if mode not in formats:
        raise ValueError('unsupported_quantization: '+(capability['rejection_reason'] or 'adapter did not validate this export format'))


def export_quantized(path,output,family,mode):
    """Explicit conversion; never copies unknown files, histories or credentials."""
    check_format(mode);path=Path(path);output=Path(output)
    if (path/MARKER).exists():raise ValueError('quantized re-encoding is not supported')
    from ..adapters.registry import inspect_model
    report=inspect_model(path,family)
    require_export_capability(report,mode)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    temp=Path(tempfile.mkdtemp(prefix='.quant-stage-',dir=output.parent))
    try:
        loader=WeightLoader(path,report,mode);entries={}
        for index,key in enumerate(report['required']):
            record=loader.read(key);name=f'tensor-{index:05d}.safetensors'
            save_file(record.parts,str(temp/name))
            original=report['entries'][key]
            entries[key]={'file':name,'shape':original['shape'],'dtype':original['dtype'],'spec':record.spec,
                'stored_bytes':sum(t.numel()*t.element_size() for t in record.parts.values())}
            del record
        loader.finish()
        for name in sorted(SIDECARS):
            if (path/name).is_file():
                if (path/name).is_symlink():raise ValueError('symlink tokenizer/config export rejected')
                shutil.copyfile(path/name,temp/name)
        files={p.name:file_hash(p) for p in sorted(temp.iterdir())}
        metadata={'format':'rvft.quantized_base.v1','family':family,'quantization':mode,
            'source_identity':loader.source_identity,'storage_identity':loader.storage_identity,'entries':entries,'files':files}
        (temp/MARKER).write_text(json.dumps(metadata,sort_keys=True)+'\n', encoding="utf-8", newline="\n")
        (temp/'COMPLETE').write_text(file_hash(temp/MARKER)+'\n', encoding="utf-8", newline="\n")
        os.rename(temp,output);temp=None
        return {'output':str(output),'quantization':mode,'source_identity':loader.source_identity,
            'storage_identity':loader.storage_identity,'stored_bytes':loader.stored_bytes,
            'max_source_tensor_bytes':loader.max_source_tensor_bytes,'full_bf16_decoder_materialized':False,
            'quantized_matrices':loader.quantized_tensors}
    finally:
        if temp is not None:shutil.rmtree(temp)
