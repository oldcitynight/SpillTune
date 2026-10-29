"""Shifted chunked CE with startup-prepared supervised chunk intervals.

The prepared training path never scans GPU labels or audits VJP contents.
One full hidden VJP is retained; one chunk of logits exists at a time.
Legacy CPU callers may derive chunk intervals before entering this function.
"""
import math
import torch
import torch.nn.functional as F

class _ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normalized, head, labels, normalization, token_chunk, supervised_chunks):
        if type(token_chunk)is not int or token_chunk<1 or not math.isfinite(normalization)or normalization<=0:
            raise ValueError('invalid terminal token chunk/normalization')
        if labels.shape!=normalized.shape[:2]or head.requires_grad:
            raise ValueError('invalid terminal labels/trainable frozen head')
        vjp=torch.zeros_like(normalized);total=torch.zeros((),device=normalized.device)
        with torch.enable_grad():
            for start,end in supervised_chunks:
                target=labels[:,start+1:end+1]
                h=normalized[:,start:end].detach().requires_grad_(True)
                logits=F.linear(h,head)
                value=F.cross_entropy(logits.float().reshape(-1,head.shape[0]),target.reshape(-1),ignore_index=-100,reduction='sum')
                gradient,=torch.autograd.grad(value/normalization,h)
                vjp[:,start:end].copy_(gradient.detach());total+=value.detach()
                del logits,value,gradient,h
        # Dynamic nonfinite loss/gradient audits are deliberately absent. The
        # execution layer reads the logged scalar once per update; full-task
        # post acceptance must still establish numerical effectiveness.
        ctx.save_for_backward(vjp)
        return total/normalization

    @staticmethod
    def backward(ctx, output_gradient):
        vjp,=ctx.saved_tensors
        return vjp*output_gradient,None,None,None,None,None

def chunked_causal_loss(normalized,head,labels,*,normalization,token_chunk,supervised_chunks=None):
    if supervised_chunks is None:
        if labels.device.type!='cpu':
            raise RuntimeError('CUDA terminal requires startup-prepared supervised chunks')
        if type(token_chunk)is not int or token_chunk<1:raise ValueError('invalid terminal token chunk')
        supervised_chunks=tuple((start,min(start+token_chunk,normalized.shape[1]-1))
            for start in range(0,normalized.shape[1]-1,token_chunk)
            if bool((labels[:,start+1:min(start+token_chunk,normalized.shape[1]-1)+1]!=-100).any()))
    return _ChunkedCE.apply(normalized,head,labels,normalization,token_chunk,supervised_chunks)
