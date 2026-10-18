"""Native E2M1/E8M0 layout; no low-bit GEMM, caching or CPU fallback.

Algorithm/layout reference: Hugging Face Transformers integrations/mxfp4.py,
Copyright 2025 The HuggingFace Team. All rights reserved. Apache-2.0; see legal/REFERENCE_NOTICES.md.
"""
import torch

VALUES=(0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.)
CHUNK_GROUPS=65536


def validate_pair(blocks,scales,*,chunk_groups=CHUNK_GROUPS):
    if blocks.dtype!=torch.uint8 or scales.dtype!=torch.uint8 or blocks.shape[:-1]!=scales.shape or blocks.shape[-1]!=16:
        raise ValueError('native_mxfp4: blocks/scales exact uint8 group32 layout required')
    if type(chunk_groups)is not int or chunk_groups<1:raise ValueError('native_mxfp4: invalid chunk size')
    b=blocks.reshape(-1,16);s=scales.reshape(-1)
    for start in range(0,s.numel(),chunk_groups):
        scale=s[start:start+chunk_groups]
        if bool((scale==255).any()):raise ValueError('native_mxfp4: E8M0 NaN exponent255')
        dangerous=scale>=253
        if bool(dangerous.any()):
            packed=b[start:start+chunk_groups][dangerous];exp=scale[dangerous]
            limit=torch.where(exp==254,4,6)[:,None]
            if bool((((packed&7)>=limit)|(((packed>>4)&7)>=limit)).any()):
                raise ValueError('native_mxfp4: BF16 overflow')


def decode_expert(blocks,scales,*,dtype=torch.bfloat16,chunk_groups=CHUNK_GROUPS):
    """[out,groups,16] -> [in,out] contiguous; BF16 ldexp BEFORE compute cast.

    Caller validates source once before staging. Chunk allocation is bounded;
    output and its transpose can coexist and are charged to adapter workspace.
    """
    if blocks.dtype!=torch.uint8 or scales.dtype!=torch.uint8 or blocks.ndim!=3 or blocks.shape[:-1]!=scales.shape or blocks.shape[-1]!=16:
        raise ValueError('native_mxfp4: invalid expert shapes')
    if dtype not in (torch.bfloat16,torch.float32) or type(chunk_groups)is not int or chunk_groups<1:
        raise ValueError('native_mxfp4: invalid compute/chunk')
    lut=torch.tensor(VALUES,device=blocks.device,dtype=torch.bfloat16)
    b=blocks.reshape(-1,16);s=scales.reshape(-1);out=torch.empty((b.shape[0],32),device=blocks.device,dtype=torch.bfloat16)
    for start in range(0,len(b),chunk_groups):
        packed=b[start:start+chunk_groups];sub=out[start:start+chunk_groups]
        idx=(packed&15).long();sub[:,0::2]=lut[idx];del idx
        idx=(packed>>4).long();sub[:,1::2]=lut[idx];del idx
        torch.ldexp(sub,(s[start:start+chunk_groups].int()-127)[:,None],out=sub)
    return out.reshape(blocks.shape[0],-1).transpose(0,1).contiguous().to(dtype)
