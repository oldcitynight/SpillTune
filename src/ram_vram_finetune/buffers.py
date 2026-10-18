"""Model-neutral raw typed buffers; interpretation belongs to a reviewed adapter.

Declared workspace is a peak budget reservation, not a preallocated scratch
buffer or a claim that arbitrary callbacks cannot exceed their declarations.
It must cover decoding in forward and autograd recomputation, including live
intermediates retained by the graph. Source contents remain bound by the
adapter identity; this contract additionally binds interpretation and layout.
"""
from dataclasses import dataclass
import re

import torch

RAW_DTYPES = frozenset((torch.uint8, torch.int8, torch.int16, torch.int32,
                       torch.int64, torch.bool, torch.float16, torch.bfloat16,
                       torch.float32, torch.float64))


@dataclass(frozen=True)
class RawBufferContract:
    decoder_id: str
    identity: str
    decode_workspace_bytes: int

    def __post_init__(self):
        if not isinstance(self.decoder_id, str) or not self.decoder_id or not self.decoder_id.isprintable():
            raise ValueError('typed_buffer: explicit decoder_id required')
        if not isinstance(self.identity, str) or not re.fullmatch('[0-9a-f]{64}', self.identity):
            raise ValueError('typed_buffer: stable SHA256 interpretation identity required')
        if type(self.decode_workspace_bytes) is not int or self.decode_workspace_bytes < 0:
            raise ValueError('typed_buffer: decode workspace must be an explicit nonnegative byte count')


def raw_buffer_metadata(block):
    """JSON values are a detached deep snapshot, never tensor/dict references."""
    contract = getattr(block, 'buffer_contract', None)
    if contract is not None and not isinstance(contract, RawBufferContract):
        raise ValueError('typed_buffer: invalid contract')
    if contract is not None and block.storage_specs is not None:
        raise ValueError('typed_buffer: raw contract cannot reinterpret an archive codec')
    if not isinstance(block.host_weights, dict) or not block.host_weights:
        raise ValueError('typed_buffer: nonempty named parts required')
    parts=[]
    for name, value in block.host_weights.items():
        if not isinstance(name,str) or not name or not isinstance(value,torch.Tensor):
            raise ValueError('typed_buffer: invalid named tensor')
        if value.dtype not in RAW_DTYPES or not value.shape or any(n < 1 for n in value.shape):
            raise ValueError('typed_buffer: unsupported dtype or empty shape')
        if contract is None and value.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('typed_buffer: opaque parts require an explicit buffer contract')
        if value.requires_grad or value.device.type!='cpu' or not value.is_contiguous():
            raise ValueError('host weights must be frozen contiguous CPU tensors')
        parts.append({'name':name,'shape':list(value.shape),'dtype':str(value.dtype),
                      'bytes':value.numel()*value.element_size()})
    return {'decoder_id':contract.decoder_id,'identity':contract.identity,
            'decode_workspace_bytes':contract.decode_workspace_bytes,'parts':parts} if contract is not None else None


def contract_bindings(blocks):
    return [{'block':b.index, **raw_buffer_metadata(b)} for b in blocks
            if getattr(b,'buffer_contract',None) is not None]
