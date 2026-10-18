"""Independent functional gpt-oss eager graph with frozen native packed experts.

Equations/operation order reference Transformers5.13 modeling_gpt_oss.py,
Copyright2025 The HuggingFace Team, Apache-2.0; see legal/REFERENCE_NOTICES.md.
"""
import torch
import torch.nn.functional as F
from .mxfp4 import decode_expert


def rms(x,weight,eps):
    value=x.float();return (weight*(value*torch.rsqrt(value.pow(2).mean(-1,keepdim=True)+eps))).to(x.dtype)


def router(x,w,top_k):
    logits=F.linear(x,w['mlp.router.weight'],w['mlp.router.bias'])
    values,indices=torch.topk(logits,top_k,dim=-1)
    return F.softmax(values,dim=1,dtype=values.dtype),indices


def feed_forward(x,w,*,experts,top_k):
    shape=x.shape;flat=x.reshape(-1,shape[-1]);scores,indices=router(flat,w,top_k)
    output=torch.zeros_like(flat)
    with torch.no_grad():
        mask=F.one_hot(indices,num_classes=experts).permute(2,1,0);hit=(mask.sum(dim=(-1,-2))>0).nonzero()
    for e in hit:
        e=e[0];position,tokens=torch.where(mask[e]);current=flat[tokens]
        gu=decode_expert(w['mlp.experts.gate_up_proj_blocks'][e],w['mlp.experts.gate_up_proj_scales'][e],dtype=x.dtype)
        gate_up=current@gu+w['mlp.experts.gate_up_proj_bias'][e]
        gate,up=gate_up[...,::2].clamp(max=7.),gate_up[...,1::2].clamp(-7.,7.)
        gated=(up+1)*(gate*torch.sigmoid(gate*1.702))
        down=decode_expert(w['mlp.experts.down_proj_blocks'][e],w['mlp.experts.down_proj_scales'][e],dtype=x.dtype)
        out=gated@down+w['mlp.experts.down_proj_bias'][e]
        output.index_add_(0,tokens,(out*scores[tokens,position,None]).to(x.dtype))
    return output.reshape(shape)


def block(hidden_states,position_embeddings,attention_mask,*,heads,kv_heads,head_dim,eps,experts,top_k,lora_q,lora_v,lora_o,**w):
    def linear(x,path,lora=None):
        weight,bias=w[path+'.weight'],w[path+'.bias']
        return lora.forward(x,weight,bias) if lora is not None else F.linear(x,weight,bias)
    residual=hidden_states;x=rms(hidden_states,w['input_layernorm.weight'],eps);shape=(*x.shape[:-1],-1,head_dim)
    q=linear(x,'self_attn.q_proj',lora_q).view(shape).transpose(1,2)
    k=linear(x,'self_attn.k_proj').view(shape).transpose(1,2)
    v=linear(x,'self_attn.v_proj',lora_v).view(shape).transpose(1,2)
    cos,sin=(t.unsqueeze(1) for t in position_embeddings)
    def rotate(t):
        a,b=t.chunk(2,dim=-1);return torch.cat((a*cos-b*sin,b*cos+a*sin),dim=-1)
    q,k=rotate(q),rotate(k)
    def repeat(t):
        b,h,s,d=t.shape
        return t if heads==kv_heads else t[:,:,None].expand(b,h,heads//kv_heads,s,d).reshape(b,heads,s,d)
    k,v=repeat(k),repeat(v);logits=(q@k.transpose(2,3))*head_dim**-.5+attention_mask
    sinks=w['self_attn.sinks'].reshape(1,-1,1,1).expand(q.shape[0],-1,q.shape[-2],-1)
    logits=torch.cat((logits,sinks),dim=-1);logits=logits-logits.max(dim=-1,keepdim=True).values
    probability=F.softmax(logits,dim=-1,dtype=logits.dtype)[...,:-1].to(v.dtype)
    attended=(probability@v).transpose(1,2).contiguous().reshape(*x.shape[:-1],heads*head_dim)
    x=residual+linear(attended,'self_attn.o_proj',lora_o);residual=x
    return residual+feed_forward(rms(x,w['post_attention_layernorm.weight'],eps),w,experts=experts,top_k=top_k)
