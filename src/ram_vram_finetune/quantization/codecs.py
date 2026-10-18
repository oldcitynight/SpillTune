"""RVFT v1 group-128 storage, NOT GGUF Q8_0/Q6_K or native low-bit GEMM.

Only storage is quantized. Decoder output is BF16; LoRA/state remain FP32.
Four offset-binary signed-six values occupy three little-endian bytes.
INT4 stores two offset-binary values per byte, low nibble first; codes
1..15 map to -7..7, zero code (-8) is reserved. This is lossy, not GGUF/NF4.
"""
from dataclasses import dataclass
import math

import torch

FORMATS = ('int4-sym-g128', 'fp8-e4m3fn-g128', 'fp8-e5m2-g128', 'int8-sym-g128', 'int6-sym-g128')
STRIDES = {'int4-sym-g128':64, 'int6-sym-g128':96}

def stride_for(mode): return STRIDES.get(mode,128)

GROUP = 128
CHUNK_GROUPS = 8192  # <=1,048,576 elements, FP32 chunk 4 MiB, not a full layer
VALIDATION_CHUNK_GROUPS = 2048  # <=262,144 decoded elements, each FP32 array <=1 MiB
DTYPES = {'bfloat16':torch.bfloat16,'float32':torch.float32}


def finite(tensor):
    flat=tensor.reshape(-1)
    return all(bool(torch.isfinite(flat[i:i+32768]).all()) for i in range(0,flat.numel(),32768))


def check_format(name):
    if name not in FORMATS:
        raise ValueError(f'unsupported_quantization: explicit format required; supported={FORMATS}; no GGUF/NF4/native-FP8 fallback')


@dataclass
class Record:
    spec: dict
    parts: dict[str, torch.Tensor]

    @property
    def shape(self): return tuple(self.spec['shape'])

    @property
    def dtype(self): return DTYPES[self.spec['dtype']]


def encoded_size(shape, source_dtype, mode, *, selected=False,raw_dtype=torch.bfloat16):
    count=math.prod(shape)
    if mode!='none' and selected:
        groups=(count+GROUP-1)//GROUP
        return groups*(stride_for(mode)+4)
    if mode!='none': return count*torch.empty((),dtype=raw_dtype).element_size()
    return count*(4 if source_dtype=='F32' else 2)


def unpack_codes(raw,mode):
    if mode.startswith('fp8'):
        return raw.view(torch.float8_e4m3fn if 'e4m3fn' in mode else torch.float8_e5m2).float()
    if mode=='int8-sym-g128':return raw.view(torch.int8).float()
    if mode=='int4-sym-g128':
        return torch.stack((raw&15,raw>>4),1).flatten().float()-8
    b=raw.view(-1,3).to(torch.int16)
    return torch.stack((b[:,0]&63,(b[:,0]>>6)|((b[:,1]&15)<<2),
        (b[:,1]>>4)|((b[:,2]&3)<<4),b[:,2]>>2),1).flatten().float()-32


def encode(tensor, mode, *, quantize=True,raw_dtype=torch.bfloat16):
    """Bounded scratch: <=CHUNK_GROUPS groups; input released by caller."""
    if tensor.device.type!='cpu' or tensor.requires_grad or not tensor.is_contiguous():
        raise ValueError('encoder requires frozen contiguous CPU weights')
    if tensor.dtype not in (torch.float32,torch.bfloat16):
        raise ValueError('unsupported_quantization: source dtype')
    spec={'version':1,'shape':list(tensor.shape),'dtype':str(tensor.dtype).split('.')[-1],'codec':'raw'}
    if mode=='none': return Record(spec,{'value':tensor})
    check_format(mode)
    if not quantize:
        dtype=raw_dtype
        value=tensor.to(dtype).contiguous()
        if not finite(value): raise ValueError('nonfinite raw weight')
        spec['dtype']=str(dtype).split('.')[-1]
        return Record(spec,{'value':value})
    count=tensor.numel();groups=(count+GROUP-1)//GROUP
    stride=stride_for(mode)
    payload=torch.empty(groups*stride,dtype=torch.uint8)
    scales=torch.empty(groups,dtype=torch.float32)
    flat=tensor.reshape(-1)
    f8={'fp8-e4m3fn-g128':torch.float8_e4m3fn,'fp8-e5m2-g128':torch.float8_e5m2}.get(mode)
    limit=torch.finfo(f8).max if f8 else ({'int4-sym-g128':7,'int6-sym-g128':31}.get(mode,127))
    for g in range(0,groups,CHUNK_GROUPS):
        end=min(g+CHUNK_GROUPS,groups);valid=min(end*GROUP,count)-g*GROUP
        chunk=torch.zeros((end-g)*GROUP,dtype=torch.float32)
        chunk[:valid].copy_(flat[g*GROUP:g*GROUP+valid]);chunk=chunk.view(-1,GROUP)
        if not torch.isfinite(chunk).all(): raise ValueError('nonfinite quantization input')
        scale=chunk.abs().amax(1)/limit
        scale=torch.where(scale==0,torch.ones_like(scale),scale)
        if not torch.isfinite(scale).all() or (scale<=0).any(): raise ValueError('invalid quantization scale')
        scales[g:end].copy_(scale)
        normalized=(chunk/scale[:,None]).clamp(-limit,limit)
        if f8:
            data=normalized.to(f8).view(torch.uint8).flatten()
        elif mode=='int8-sym-g128':
            data=normalized.round().to(torch.int8).view(torch.uint8).flatten()
        elif mode=='int4-sym-g128':
            q=(normalized.round().to(torch.uint8)+8).flatten().view(-1,2)
            data=(q[:,0]|(q[:,1]<<4)).flatten()
        else:
            q=(normalized.round().to(torch.int16)+32).flatten().view(-1,4)
            data=torch.stack((q[:,0]|((q[:,1]&3)<<6), (q[:,1]>>2)|((q[:,2]&15)<<4),
                              (q[:,2]>>4)|(q[:,3]<<2)),1).to(torch.uint8).flatten()
        payload[g*stride:end*stride].copy_(data)
        decoded=unpack_codes(data,mode).view(-1,GROUP)*scale[:,None]
        # A finite BF16 conversion also proves its FP32 input was finite:
        # NaN/Inf cannot convert to finite BF16. One full scan rejects both.
        if not torch.isfinite(decoded.to(torch.bfloat16)).all():
            raise ValueError('quantized value overflows BF16 compute range')
    spec.update(dtype='bfloat16',codec=mode,group_size=GROUP,numel=count)
    return Record(spec,{'payload':payload,'scales':scales})


def validate_spec(spec, parts):
    if spec.get('version')!=1 or not isinstance(spec.get('shape'),list) or not spec['shape'] or any(type(n)is not int or n<1 for n in spec['shape']):
        raise ValueError('invalid stored weight shape/version')
    if spec.get('dtype') not in DTYPES: raise ValueError('invalid stored compute dtype')
    count=math.prod(spec['shape'])
    if spec.get('codec')=='raw':
        if set(spec)!= {'version','shape','dtype','codec'} or set(parts)!= {'value'} or tuple(parts['value'].shape)!=tuple(spec['shape']) or parts['value'].dtype!=DTYPES[spec['dtype']]:
            raise ValueError('invalid raw storage schema')
    else:
        check_format(spec.get('codec'))
        if set(spec)!={'version','shape','dtype','codec','group_size','numel'} or spec['group_size']!=GROUP or spec['numel']!=count or spec['dtype']!='bfloat16':
            raise ValueError('invalid quantized storage schema')
        groups=(count+GROUP-1)//GROUP;stride=stride_for(spec['codec'])
        if set(parts)!= {'payload','scales'} or parts['payload'].dtype!=torch.uint8 or tuple(parts['payload'].shape)!=(groups*stride,) or parts['scales'].dtype!=torch.float32 or tuple(parts['scales'].shape)!=(groups,):
            raise ValueError('invalid quantized part inventory/shape/dtype')
        if not torch.isfinite(parts['scales']).all() or (parts['scales']<=0).any(): raise ValueError('invalid stored scales')
        payload=parts['payload'];mode=spec['codec']
        for first_group in range(0,groups,VALIDATION_CHUNK_GROUPS):
            end=min(first_group+VALIDATION_CHUNK_GROUPS,groups)
            raw=payload[first_group*stride:end*stride]
            codes=unpack_codes(raw,mode)
            # Integer bit layouts always produce exactly representable bounded
            # FP32 codes; FP8 additionally has nonfinite bit patterns.
            if mode.startswith('fp8') and not torch.isfinite(codes).all():raise ValueError('nonfinite FP8 code')
            if mode=='int4-sym-g128' and (codes==-8).any():raise ValueError('reserved int4 -8 code')
            if mode=='int8-sym-g128' and (codes==-128).any():raise ValueError('reserved int8 -128 code')
            if mode=='int6-sym-g128' and (codes==-32).any():raise ValueError('reserved int6 -32 code')
            chunk_groups=codes.numel()//GROUP
            valid=max(0,min(codes.numel(),count-first_group*GROUP))
            if (codes[valid:]!=0).any():raise ValueError('nonzero quantized tail padding')
            decoded=codes.view(-1,GROUP)*parts['scales'][first_group:first_group+chunk_groups,None]
            # This full conversion/finite predicate rejects both FP32 product
            # nonfinites and finite FP32 values overflowing BF16.
            if not torch.isfinite(decoded.to(torch.bfloat16)).all():
                raise ValueError('quantized value overflows BF16 compute range')
    if spec['codec']=='raw' and not finite(parts['value']): raise ValueError('nonfinite raw weight')
    if any(t.requires_grad or not t.is_contiguous() for t in parts.values()): raise ValueError('invalid trainable/noncontiguous frozen storage')


def decode_into(destination, spec, parts):
    """Use caller-owned BF16 slot and bounded CHUNK_GROUPS temporaries."""
    if spec['codec']=='raw':
        destination.copy_(parts['value']);return
    count=spec['numel'];groups=(count+GROUP-1)//GROUP
    mode=spec['codec'];stride=stride_for(mode)
    flat=destination.reshape(-1)
    for g in range(0,groups,CHUNK_GROUPS):
        end=min(g+CHUNK_GROUPS,groups);raw=parts['payload'][g*stride:end*stride]
        values=unpack_codes(raw,mode)
        values=values.view(-1,GROUP)*parts['scales'][g:end,None]
        valid=min(end*GROUP,count)-g*GROUP
        flat[g*GROUP:g*GROUP+valid].copy_(values.flatten()[:valid])


def materialize(record, device='cpu'):
    if record.spec['codec']=='raw':return record.parts['value'].to(device)
    value=torch.empty(record.shape,dtype=record.dtype,device=device)
    decode_into(value,record.spec,{k:t.to(device) for k,t in record.parts.items()})
    return value


def flatten_records(records):
    tensors,specs={},{}
    for name,record in records.items():
        specs[name]={'spec':record.spec,'parts':{}}
        for part,tensor in record.parts.items():
            key=name if record.spec['codec']=='raw' else f'@{name}/{part}'
            if key in tensors: raise ValueError('storage name collision')
            tensors[key]=tensor;specs[name]['parts'][part]=key
    return tensors,specs
