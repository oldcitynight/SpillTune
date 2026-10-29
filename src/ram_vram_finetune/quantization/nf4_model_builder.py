"""Prepared per-tensor Qwen3.5 converter. No conversion occurs on import.

This code is unqualified for real-model conversion until a separate bounded
conversion job measures its peak memory. Geometry limits are not RAM guarantees.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path

ITEM_BYTES={'BF16':2,'F32':4}

def conversion_geometry(report):
    required=report.get('required');entries=report.get('entries');selected=set(report.get('quantization_matrix_keys',()))
    if type(required)is not list or not required or len(set(required))!=len(required) or type(entries)is not dict:
        raise ValueError('invalid source header inventory')
    source=canonical=largest=0
    for key in required:
        entry=entries[key];shape=entry['shape'];dtype=entry['dtype']
        if dtype not in ITEM_BYTES or not shape or any(type(v)is not int or v<=0 for v in shape):raise ValueError('unsupported source tensor metadata')
        n=math.prod(shape);raw=n*ITEM_BYTES[dtype];source+=raw;largest=max(largest,raw)
        if key in selected:
            if len(shape)!=2 or dtype!='BF16':raise ValueError('NF4 eligible source must be BF16 matrix')
            blocks=(n+63)//64
            canonical+=(n+1)//2+blocks+4*((blocks+255)//256)+64+1024+4
        else:
            canonical+=n*(4 if key.endswith(('.linear_attn.A_log','.linear_attn.norm.weight','.linear_attn.dt_bias')) else ITEM_BYTES[dtype])
    return {'required_source_bytes':source,'largest_source_tensor_bytes':largest,'canonical_payload_lower_bound_bytes':canonical,'source_tensor_count':len(required),'NF4_matrix_count':len(selected),'peak_RAM_qualified':False,'note':'Excludes allocator/quantizer/serialization scratch, constructor clones, native model copies, caches and OS pages; never treat as measured peak.'}

def _limits(limits):
    keys={'max_source_bytes','max_source_tensor_bytes','max_canonical_payload_bytes'}
    if type(limits)is not dict or set(limits)!=keys or any(type(v)is not int or v<=0 for v in limits.values()):
        raise ValueError('explicit positive geometry limits required; real RAM limit belongs to owning cgroup')
    return limits

def prepare_conversion(source_path,*,limits):
    """Header-only inspection. Does not open tensor payloads or quantize."""
    limits=_limits(limits)
    from ..adapters.qwen35.adapter import inspect_model
    report=inspect_model(Path(source_path))
    if report.get('architecture')!='qwen35' or report['text_config'].get('tie_word_embeddings')is not True:
        raise ValueError('only original tied Qwen3.5 text source supported')
    if (Path(source_path)/'rvft_quantization.json').exists():raise ValueError('Q5/other archives cannot substitute for original BF16 source')
    if report.get('source_aliases')!={'lm_head.weight':'model.language_model.embed_tokens.weight'}:raise ValueError('canonical tied source alias required before payload')
    from ..adapters.qwen35.nf4_cpu import _source_inventory
    _source_inventory(report)  # Exact source role dtypes, no old 16MiB cap change.
    geometry=conversion_geometry(report)
    for observed,limit in [('required_source_bytes','max_source_bytes'),('largest_source_tensor_bytes','max_source_tensor_bytes'),('canonical_payload_lower_bound_bytes','max_canonical_payload_bytes')]:
        if geometry[observed]>limits[limit]:raise MemoryError('conversion geometry exceeds explicit '+limit)
    return report,geometry

def require_conversion_admission(guard,stage,geometry):
    if guard(stage,dict(geometry)) is not True:
        raise RuntimeError('conversion resource guard did not explicitly admit phase '+stage)

def build_profiled_qwen35_nf4_artifact(source_path,*,limits,conversion_guard):
    """Convert one tensor at a time under an externally enforced resource job.

    Not executed in the CPU preparation batch. The guard must check the owning
    cgroup/resources and may raise before each phase. A failed job publishes no
    model artifact; callers use save_nf4_model only after this returns complete.
    """
    _limits(limits)
    if not callable(conversion_guard):raise TypeError('explicit conversion resource guard required')
    report,geometry=prepare_conversion(source_path,limits=limits)
    def check(stage):
        require_conversion_admission(conversion_guard,stage,geometry)
    check('before_payload')
    import torch
    from safetensors import safe_open
    from ..quantization.nf4_state import NF4Record
    from ..quantization.nf4_model_state import NF4ModelArtifact
    from ..adapters.qwen35.nf4_cpu import _canonical_json,_FP32_SOURCE_SUFFIXES,_FP32_ARTIFACT_SUFFIXES,_SOURCE_SCHEMA,_SOURCE_KIND,_PREFIX
    digest=hashlib.sha256(b'rvft.nf4_source_payload.v1\0');records={};tensors={};parameters=0
    selected=set(report['quantization_matrix_keys'])
    for key in sorted(report['required']):
        check('before_tensor:'+key)
        entry=report['entries'][key]
        with safe_open(entry['file'],framework='pt',device='cpu') as handle:
            value=handle.get_tensor(key).detach().contiguous()
        expected=torch.float32 if key.endswith(_FP32_SOURCE_SUFFIXES) else torch.bfloat16
        if value.dtype is not expected or list(value.shape)!=entry['shape']:raise ValueError('source role dtype/header drift: '+key)
        flat=value.reshape(-1)
        for start in range(0,flat.numel(),1024**2):
            check('finite_chunk:'+key)
            if not bool(torch.isfinite(flat[start:start+1024**2]).all()):raise ValueError('nonfinite source tensor: '+key)
        del flat
        descriptor=_canonical_json({'key':key,'dtype':str(value.dtype).removeprefix('torch.'),'shape':list(value.shape)})
        raw=memoryview(value.view(torch.uint8).reshape(-1).numpy()).cast('B')
        digest.update(len(descriptor).to_bytes(8,'big'));digest.update(descriptor);digest.update(len(raw).to_bytes(8,'big'))
        for start in range(0,len(raw),8*1024**2):
            check('hash_chunk:'+key);digest.update(raw[start:start+8*1024**2])
        del raw
        parameters+=value.numel()
        if key in selected:records[key]=NF4Record.from_weight(value)
        else:tensors[key]=value.to(dtype=torch.float32) if key.endswith(_FP32_ARTIFACT_SUFFIXES) else value
        del value;check('after_tensor:'+key)
    source={'schema':_SOURCE_SCHEMA,'kind':_SOURCE_KIND,'config_sha256':hashlib.sha256(_canonical_json(report['text_config'])).hexdigest(),'payload_sha256':digest.hexdigest(),'text_tensor_count':len(report['required']),'text_parameter_count':parameters,'excluded_tensors':{k:len(report.get('excluded_nontext_tensors',{}).get(k,())) for k in ('vision','mtp')}}
    if report.get('source_aliases')!={'lm_head.weight':_PREFIX+'embed_tokens.weight'}:raise ValueError('canonical tied alias required')
    check('before_canonical_ownership_copy')
    artifact=NF4ModelArtifact.from_parts(config=report['text_config'],records=records,tensors=tensors,aliases=report['source_aliases'],source=source)
    del records,tensors
    check('complete')
    return artifact
