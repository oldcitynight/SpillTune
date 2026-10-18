"""Native DeepSeek-V3 decoder subset: MLA, grouped MoE and frozen routers.

Independent functional graph against Transformers 5.13.0 contracts. No native
decoder modules/weights retained on device; no remote code or model aliases.
"""
from ..quantization.codecs import FORMATS

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .dense import rms
from .protocol import BlockPlan
from ..finetune.lora import LoRAProjection
from ..quantization.storage import scan_entries,WeightLoader,estimated_bytes
from ..quantization.codecs import materialize,flatten_records,finite


def hf_classes():
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3DecoderLayer,DeepseekV3ForCausalLM
    return DeepseekV3Config,DeepseekV3DecoderLayer,DeepseekV3ForCausalLM


def inspect_deepseek(path):
    root=Path(path);raw=json.loads((root/'config.json').read_text(encoding="utf-8"))
    def reject(reason):raise ValueError('unsupported_deepseek: '+reason)
    if raw.get('model_type')!='deepseek_v3':reject('native deepseek_v3 only; V2/V32/V4 and Qwen/Llama distill aliases require distinct adapters')
    if raw.get('quantization_config') is not None:reject('external FP8/quantization_config is not RVFT g128; explicit BF16/F32 source required')
    if any(raw.get(k,0) not in (None,0) for k in ['num_nextn_predict_layers','num_nextn_layers','num_mtp_layers','mtp_num_hidden_layers']):reject('MTP tensors/objective')
    if any(k.startswith(('index_','hc_','compress_')) for k in raw):reject('DSA/mHC/compressed attention is not V3 MLA')
    if any(raw.get(k,0) not in (None,0) for k in ['aux_loss_alpha','router_aux_loss_coef']):reject('auxiliary router training objective; CE-only frozen router supported')
    if raw.get('ep_size',1)!=1 or raw.get('pretraining_tp',1)!=1:reject('expert/tensor parallel execution')
    if raw.get('moe_layer_freq',1)!=1:reject('noncontiguous MoE layer schedule')
    if raw.get('scoring_func','sigmoid')!='sigmoid' or raw.get('topk_method','noaux_tc')!='noaux_tc':reject('only sigmoid/noaux_tc grouped router')
    if raw.get('sliding_window') is not None or raw.get('use_sliding_window',False):reject('sliding window')
    Config,Layer,_=hf_classes();config=Config(**raw);config._attn_implementation='eager';config._experts_implementation='eager'
    if config.tie_word_embeddings or config.attention_bias or raw.get('mlp_bias',False):reject('shared embeddings or projection biases')
    if config.hidden_act!='silu' or config.attention_dropout!=0:reject('activation/dropout')
    positive=['hidden_size','vocab_size','intermediate_size','moe_intermediate_size','num_hidden_layers','num_attention_heads',
        'num_key_value_heads','kv_lora_rank','qk_nope_head_dim','qk_rope_head_dim','v_head_dim','n_routed_experts',
        'n_shared_experts','n_group','topk_group','num_experts_per_tok','max_position_embeddings']
    if any(type(getattr(config,k))is not int or getattr(config,k)<1 for k in positive):reject('positive integer dimensions required')
    if config.q_lora_rank is not None and (type(config.q_lora_rank)is not int or config.q_lora_rank<1):reject('invalid q_lora_rank')
    if config.num_key_value_heads!=config.num_attention_heads or config.qk_rope_head_dim%2:reject('MLA head grouping/rotary dimensions')
    if type(config.first_k_dense_replace)is not int or not 0<=config.first_k_dense_replace<config.num_hidden_layers:reject('at least one actual MoE layer required')
    if type(config.norm_topk_prob)is not bool or type(config.rope_interleave)is not bool:reject('router/rotary booleans')
    if config.n_routed_experts%config.n_group or config.n_routed_experts//config.n_group<2:reject('router groups need >=2 experts each')
    if config.topk_group>config.n_group or config.num_experts_per_tok>config.topk_group*(config.n_routed_experts//config.n_group):reject('top-k exceeds selected group capacity')
    if not all(math.isfinite(x) and x>0 for x in [config.rms_norm_eps,config.routed_scaling_factor]):reject('invalid epsilon/router scale')
    rope=config.rope_parameters
    allowed={'rope_type','rope_theta'}
    if rope.get('rope_type')=='yarn':allowed|={'factor','original_max_position_embeddings','beta_fast','beta_slow','mscale','mscale_all_dim','attention_factor','truncate'}
    elif rope.get('rope_type')!='default':reject('only default/YaRN rotary embeddings')
    if set(rope)-allowed or not math.isfinite(rope['rope_theta']) or rope['rope_theta']<=1:reject('unsupported rotary parameters')
    if rope['rope_type']=='yarn':
        if not all(type(rope.get(k))in (int,float) and math.isfinite(rope[k]) and rope[k]>0 for k in ['factor','original_max_position_embeddings']):reject('invalid YaRN factor/context')
        if any(rope.get(k) is not None and (type(rope[k])not in (int,float) or not math.isfinite(rope[k]) or rope[k]<=0) for k in ['beta_fast','beta_slow','mscale','mscale_all_dim','attention_factor']):reject('invalid YaRN scaling')
        if type(rope.get('truncate',True))is not bool:reject('invalid YaRN truncate')
    with torch.device('meta'):
        shapes={f'model.layers.{i}.{k}':list(v.shape) for i in range(config.num_hidden_layers) for k,v in Layer(config,i).state_dict().items()}
    shapes.update({'model.embed_tokens.weight':[config.vocab_size,config.hidden_size],
        'model.norm.weight':[config.hidden_size],'lm_head.weight':[config.vocab_size,config.hidden_size]})
    entries=scan_entries(root,'deepseek_v3')
    first=f'model.layers.{config.first_k_dense_replace}.mlp.experts.'
    packed=first+'gate_up_proj' in entries
    if not packed:
        for i in range(config.first_k_dense_replace,config.num_hidden_layers):
            prefix=f'model.layers.{i}.mlp.experts.'
            shapes.pop(prefix+'gate_up_proj');shapes.pop(prefix+'down_proj')
            for e in range(config.n_routed_experts):
                shapes.update({prefix+f'{e}.{p}.weight':shape for p,shape in [
                    ('gate_proj',[config.moe_intermediate_size,config.hidden_size]),
                    ('up_proj',[config.moe_intermediate_size,config.hidden_size]),
                    ('down_proj',[config.hidden_size,config.moe_intermediate_size])]})
    if set(entries)!=set(shapes):reject(f'checkpoint inventory/layout; missing={sorted(shapes.keys()-entries.keys())}; extra={sorted(entries.keys()-shapes.keys())}')
    if any(entries[k]['shape']!=s for k,s in shapes.items()):reject('checkpoint/config shape mismatch')
    router=[k for k in entries if k.endswith(('.mlp.gate.weight','.mlp.gate.e_score_correction_bias'))]
    dtype=entries['model.embed_tokens.weight']['dtype']
    if dtype not in {'F32','BF16'} or any(v['dtype'] not in ({dtype,'F32'} if k in router else {dtype}) for k,v in entries.items()):reject('BF16/F32 source with homogeneous compute weights and optional F32 routers required')
    itemsize=4 if dtype=='F32' else 2
    return {'quantized_export': {'version':1,'formats':list(FORMATS),'rejection_reason':''}, 'architecture':'deepseek_v3','layers':config.num_hidden_layers,'config':config,'text_config':raw,
        'entries':entries,'required':sorted(entries),'expert_layout':'packed' if packed else 'individual',
        'quantization_matrix_keys':[k for k,v in entries.items() if k.startswith('model.layers.') and k not in router and len(v['shape'])in (2,3) and math.prod(v['shape'])>=128],
        'quantization_fp32_keys':router,'head_bytes':entries['lm_head.weight'].get('stored_bytes',config.vocab_size*config.hidden_size*itemsize)}


def mscale(factor,scale=1.):return 1. if factor<=1 else 1.+.1*scale*math.log(factor)


def rope_parameters(config,device):
    p=config.rope_parameters;dim=config.qk_rope_head_dim;base=p['rope_theta']
    frequencies=base**(torch.arange(0,dim,2,device=device,dtype=torch.float32)/dim)
    if p['rope_type']=='default':return 1./frequencies,1.
    factor=p['factor'];original=p['original_max_position_embeddings']
    def correction(rot):return dim*math.log(original/(rot*2*math.pi))/(2*math.log(base))
    lo=correction(p.get('beta_fast',32) or 32);hi=correction(p.get('beta_slow',1) or 1)
    if p.get('truncate',True):lo=math.floor(lo);hi=math.ceil(hi)
    lo=max(lo,0);hi=min(hi,dim-1)
    if lo==hi:hi+=.001
    ramp=((torch.arange(dim//2,device=device,dtype=torch.float32)-lo)/(hi-lo)).clamp(0,1)
    # Preserve two subtractions: algebraic cancellation changes long-context
    # frequencies in FP32, even though the real-valued formula is identical.
    extrapolation=1-ramp
    inverse=(1./(factor*frequencies))*(1-extrapolation)+(1./frequencies)*extrapolation
    amplitude=p.get('attention_factor')
    if amplitude is None:
        amplitude=mscale(factor,p['mscale'])/mscale(factor,p['mscale_all_dim']) if p.get('mscale') and p.get('mscale_all_dim') else mscale(factor)
    return inverse,amplitude


def route(hidden,weight,bias,*,groups,group_k,top_k,normalize,scale):
    """Frozen F32 router; bias changes selection, never selected probability."""
    logits=F.linear(hidden.reshape(-1,hidden.shape[-1]).float(),weight.float())
    scores=logits.sigmoid();choice=scores+bias.float()
    group_scores=choice.view(-1,groups,choice.shape[-1]//groups).topk(2,dim=-1)[0].sum(-1)
    selected_groups=group_scores.topk(group_k,dim=-1,sorted=False)[1]
    mask=torch.zeros_like(group_scores).scatter_(1,selected_groups,1).bool()
    mask=mask[:,:,None].expand(-1,groups,choice.shape[-1]//groups).reshape_as(choice)
    indices=choice.masked_fill(~mask,float('-inf')).topk(top_k,dim=-1,sorted=False)[1]
    weights=scores.gather(1,indices)
    if normalize:weights=weights/(weights.sum(-1,keepdim=True)+1e-20)
    return logits,weights*scale,indices


def deepseek_block(hidden_states,position_embeddings,attention_mask,*,parameters,
                   lora_q=None,lora_qa=None,lora_qb=None,lora_kva=None,lora_kvb=None,lora_o=None,**w):
    c=parameters
    def linear(x,name,lora=None):
        weight=w[name+'.weight'];return F.linear(x,weight) if lora is None else lora.forward(x,weight)
    residual=hidden_states;x=rms(hidden_states,w['input_layernorm.weight'],c['eps'])
    q=linear(x,'self_attn.q_proj',lora_q) if c['q_rank'] is None else linear(
        rms(linear(x,'self_attn.q_a_proj',lora_qa),w['self_attn.q_a_layernorm.weight'],1e-6),'self_attn.q_b_proj',lora_qb)
    q=q.view(*x.shape[:-1],c['heads'],c['nope']+c['rope']).transpose(1,2)
    qp,qr=q.split([c['nope'],c['rope']],-1)
    compressed=linear(x,'self_attn.kv_a_proj_with_mqa',lora_kva)
    kv,kr=compressed.split([c['kv_rank'],c['rope']],-1)
    kv=linear(rms(kv,w['self_attn.kv_a_layernorm.weight'],1e-6),'self_attn.kv_b_proj',lora_kvb)
    kv=kv.view(*x.shape[:-1],c['heads'],c['nope']+c['value']).transpose(1,2)
    kp,value=kv.split([c['nope'],c['value']],-1);kr=kr.view(x.shape[0],1,x.shape[1],c['rope'])
    cos,sin=(a.unsqueeze(1) for a in position_embeddings)
    def rotate(t):
        if c['interleave']:
            a,b=t[...,0::2],t[...,1::2];co,si=cos[...,:c['rope']//2],sin[...,:c['rope']//2]
            return torch.cat((a*co-b*si,b*co+a*si),-1)
        a,b=t.chunk(2,-1);return t*cos+torch.cat((-b,a),-1)*sin
    query=torch.cat((qp,rotate(qr)),-1);key=torch.cat((kp,rotate(kr).expand_as(qr)),-1)
    scores=(query@key.transpose(-2,-1))*c['attention_scale']
    probabilities=torch.softmax(scores+attention_mask,-1,dtype=torch.float32).to(query.dtype)
    attended=(probabilities@value).transpose(1,2).contiguous().reshape(*x.shape[:-1],c['heads']*c['value'])
    x=residual+linear(attended,'self_attn.o_proj',lora_o);residual=x
    x=rms(x,w['post_attention_layernorm.weight'],c['eps'])
    def mlp(t,prefix):return linear(F.silu(linear(t,prefix+'gate_proj'))*linear(t,prefix+'up_proj'),prefix+'down_proj')
    if not c['moe']:return residual+mlp(x,'mlp.')
    flat=x.reshape(-1,x.shape[-1]);out=torch.zeros_like(flat)
    _,weights,indices=route(flat,w['mlp.gate.weight'],w['mlp.gate.e_score_correction_bias'],
        groups=c['groups'],group_k=c['group_k'],top_k=c['top_k'],normalize=c['normalize'],scale=c['router_scale'])
    with torch.no_grad():
        mask=F.one_hot(indices,num_classes=c['experts']).permute(2,1,0)
        hits=(mask.sum((-1,-2))>0).nonzero()
    for expert in hits:
        expert=expert[0];top_position,token=torch.where(mask[expert]);current=flat[token]
        if c['packed']:
            gate,up=F.linear(current,w['mlp.experts.gate_up_proj'][expert]).chunk(2,-1)
            current=F.linear(F.silu(gate)*up,w['mlp.experts.down_proj'][expert])
        else:
            prefix=f'mlp.experts.{int(expert)}.'
            joined=torch.cat((w[prefix+'gate_proj.weight'],w[prefix+'up_proj.weight']),0)
            gate,up=F.linear(current,joined).chunk(2,-1)
            current=F.linear(F.silu(gate)*up,w[prefix+'down_proj.weight'])
        current=current*weights[token,top_position,None]
        out.index_add_(0,token,current.to(out.dtype))
    return residual+(out.view_as(x)+mlp(x,'mlp.shared_experts.'))


class DeepseekV3Adapter:
    def __init__(self,path,*,rank=2,alpha=4.,device='cpu',host_budget_bytes=1024**3,device_budget_bytes=2*1024**3,
                 quantization='none',quantized_compute='bf16'):
        report=inspect_deepseek(path)
        if quantized_compute!='bf16':raise ValueError('unsupported_quantization: only storage plus BF16 compute')
        loader=WeightLoader(path,report,quantization);boundary={'lm_head.weight','model.norm.weight'}
        if sum(estimated_bytes(report,quantization,k) for k in report['required'] if k not in boundary)>host_budget_bytes or estimated_bytes(report,quantization,'lm_head.weight')>device_budget_bytes:
            raise MemoryError('resource_budget: model exceeds host/head allowance before loading')
        # Conservative aligned slot + upload + boundary check, BEFORE reading
        # any payload. Every expert of a layer counts, not only active top-k.
        slot=upload=0
        for i in range(report['layers']):
            keys=[k for k in report['required'] if k.startswith(f'model.layers.{i}.')]
            decoded=sum(math.prod(report['entries'][k]['shape'])*(4 if (quantization=='none' and report['entries'][k]['dtype']=='F32') or (quantization!='none' and k in report['quantization_fp32_keys']) else 2)+255 for k in keys)
            slot=max(slot,decoded)
            if quantization!='none':upload=max(upload,sum(estimated_bytes(report,quantization,k)+511 for k in keys))
        if slot+upload+sum(estimated_bytes(report,quantization,k) for k in boundary)>device_budget_bytes:
            raise MemoryError('resource_budget: full layer including all experts/decoded+upload slots and boundaries exceeds device budget before loading')
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('dependency: explicit CUDA requested but unavailable')
        self.config=report['config'];self.layout=report['expert_layout'];state={}
        for k in report['required']:
            r=loader.read(k);state[k]=materialize(r,self.device) if k in boundary else r
            if k in report['quantization_fp32_keys'] and not all(finite(t) for t in r.parts.values()):raise ValueError('unsupported_deepseek: nonfinite router')
        self.identity=loader.finish();self.embedding=materialize(state.pop('model.embed_tokens.weight'))
        self.norm=state.pop('model.norm.weight');self.head=state.pop('lm_head.weight');self.blocks=[]
        c=self.config;scale=(c.qk_nope_head_dim+c.qk_rope_head_dim)**-.5
        if c.rope_parameters['rope_type']=='yarn' and c.rope_parameters.get('mscale_all_dim',0):scale*=mscale(c.rope_parameters['factor'],c.rope_parameters['mscale_all_dim'])**2
        self.targets=(['self_attn.q_proj'] if c.q_lora_rank is None else ['self_attn.q_a_proj','self_attn.q_b_proj'])+[
            'self_attn.kv_a_proj_with_mqa','self_attn.kv_b_proj','self_attn.o_proj']
        for i in range(c.num_hidden_layers):
            prefix=f'model.layers.{i}.';records={k.removeprefix(prefix):state.pop(k) for k in list(state) if k.startswith(prefix)}
            adapters={}
            for name in self.targets:
                out_features,in_features=records[name+'.weight'].shape
                a=torch.randn(rank,in_features,device=self.device,dtype=torch.float32)*.02;b=torch.zeros(out_features,rank,device=self.device)
                adapters[name]=LoRAProjection(name=name,rank=rank,alpha=alpha,in_features=in_features,out_features=out_features,
                    A=a,B=b,grad_A=torch.zeros_like(a),grad_B=torch.zeros_like(b),m_A=torch.zeros_like(a),v_A=torch.zeros_like(a),
                    m_B=torch.zeros_like(b),v_B=torch.zeros_like(b),layer=i,target_path=name,architecture='deepseek_v3')
            parameters={'heads':c.num_attention_heads,'nope':c.qk_nope_head_dim,'rope':c.qk_rope_head_dim,'value':c.v_head_dim,
                'kv_rank':c.kv_lora_rank,'q_rank':c.q_lora_rank,'eps':c.rms_norm_eps,'attention_scale':scale,
                'interleave':c.rope_interleave,'moe':i>=c.first_k_dense_replace,'packed':self.layout=='packed',
                'experts':c.n_routed_experts,'groups':c.n_group,'group_k':c.topk_group,'top_k':c.num_experts_per_tok,
                'normalize':c.norm_topk_prob,'router_scale':c.routed_scaling_factor}
            names=['lora_q'] if c.q_lora_rank is None else ['lora_qa','lora_qb']
            kwargs={'parameters':parameters,**dict(zip(names+['lora_kva','lora_kvb','lora_o'],adapters.values()))}
            tensors,specs=flatten_records(records)
            self.blocks.append(BlockPlan(i,'mla_moe',deepseek_block,kwargs,tensors,adapters,specs if quantization!='none' else None))
        self.storage_metadata=None if quantization=='none' else {'format':quantization,'compute':'bf16','storage_identity':loader.storage_identity,
            'source_identity':loader.source_identity,'total_stored_bytes':loader.stored_bytes,'max_source_tensor_bytes':loader.max_source_tensor_bytes,
            'full_bf16_decoder_retained':False,'quantized_scope':'MLA/MLP/shared/routed expert matrices, including packed 3D; F32 router excluded',
            'quantized_matrices':loader.quantized_tensors,'decoder_stored_bytes':sum(t.numel()*t.element_size() for b in self.blocks for t in b.host_weights.values()),
            'cpu_embedding_bytes':self.embedding.numel()*self.embedding.element_size(),
            'device_boundary_bytes':self.head.numel()*self.head.element_size()+self.norm.numel()*self.norm.element_size(),'resident_decoder_core_bytes':0}

    def positions_and_masks(self,valid):
        if valid.dtype!=torch.bool or valid.ndim!=2 or not 1<=valid.shape[1]<=self.config.max_position_embeddings:raise ValueError('unsupported_deepseek: invalid mask/context')
        if not valid[:,0].all() or ((~valid[:,:-1])&valid[:,1:]).any():raise ValueError('unsupported_deepseek: all-valid/right padding only')
        valid=valid.to(self.device);inverse,amplitude=rope_parameters(self.config,self.device)
        pos=torch.arange(valid.shape[1],device=self.device).expand(valid.shape[0],-1)
        phase=(inverse[None,:,None].expand(valid.shape[0],-1,1)@pos[:,None].float()).transpose(1,2);phase=torch.cat((phase,phase),-1)
        rotary=((phase.cos()*amplitude).to(self.head.dtype),(phase.sin()*amplitude).to(self.head.dtype))
        blocked=torch.ones(valid.shape[1],valid.shape[1],device=self.device,dtype=torch.bool).triu(1)[None,None]|~valid[:,None,None,:]
        mask=torch.zeros(blocked.shape,device=self.device,dtype=self.head.dtype).masked_fill(blocked,torch.finfo(self.head.dtype).min)
        return rotary,{'mla_moe':mask}

    def terminal_norm(self,x):return rms(x,self.norm,self.config.rms_norm_eps)
    def artifact_key(self,layer,target,parameter):return f'base_model.model.model.layers.{layer}.{target}.lora_{parameter}.weight'
    def runtime_metadata(self):return {'quantized_export': {'version':1,'formats':list(FORMATS),'rejection_reason':''}, 'architecture':'deepseek_v3','layer_count':len(self.blocks),'expert_layout':self.layout,
        'attention':'eager_MLA','expert_backend':'eager_per_expert_linear','rope':self.config.rope_parameters,'router':'frozen F32 sigmoid/noaux_tc; group top2 scores; correction only selection',
        'lora_targets':self.targets,'expert_training':'frozen routed/shared experts; input gradients propagate','objective':'causal_CE_only_no_aux_no_MTP',
        'kv_cache':'disabled','asynchronous_transfers':'unsupported','expert_scheduler':'all experts of one layer share one slot'}
    def close(self):self.blocks.clear();self.embedding=self.head=self.norm=None
