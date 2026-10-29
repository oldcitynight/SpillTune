"""Explicit shared initial LoRA state, independent of global RNG and arm order."""
import math
import torch

def make_canonical_lora(projections, *, seed=0, b_std=0.):
    if type(seed) is not int or seed<0 or not math.isfinite(b_std) or b_std<0:
        raise ValueError('invalid canonical LoRA initialization')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    return {key:{'A':torch.randn(tuple(p.A.shape),generator=generator,dtype=torch.float32)*.02,
                 'B':torch.randn(tuple(p.B.shape),generator=generator,dtype=torch.float32)*b_std,
                 'scale':float(p.scale)} for key,p in sorted(projections.items())}

def load_canonical_lora(projections, state):
    if set(state)!=set(projections):raise ValueError('canonical LoRA target inventory mismatch')
    for key,p in projections.items():
        item=state[key]
        if set(item)!={'A','B','scale'} or item['scale']!=float(p.scale):raise ValueError('canonical LoRA scale/schema mismatch')
        for field in ('A','B'):
            value=item[field]
            if value.dtype!=torch.float32 or value.shape!=getattr(p,field).shape or not torch.isfinite(value).all():
                raise ValueError('canonical LoRA shape/dtype/value mismatch')
    with torch.no_grad():
        for key,p in projections.items():
            for field in ('A','B'):getattr(p,field).copy_(state[key][field])


def canonical_fingerprint(state):
    import hashlib,json
    digest=hashlib.sha256()
    for key,item in sorted(state.items()):
        digest.update(json.dumps({'target':key,'scale':item['scale']},sort_keys=True).encode())
        for field in ('A','B'):
            tensor=item[field].detach().cpu().contiguous()
            digest.update(json.dumps({'field':field,'shape':list(tensor.shape),'dtype':str(tensor.dtype)},sort_keys=True).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
