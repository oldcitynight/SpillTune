"""Native Qwen3-MoE CE-only attention LoRA, all-expert whole-layer slot.

Canonical HF per-expert2D or native packed3D sources are explicit layouts. The
router/experts stay frozen but remain in the input VJP. No expert prefetch,
auxiliary objective, MTP, compressed source or trainable shared boundary.
"""
from ..quantization.codecs import FORMATS

import functools
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from .dense import DenseDecoderAdapter,dense_block
from .protocol import BlockPlan
from ..finetune.lora import create_lora_for_block
from .targets import ATTENTION_TARGETS
from ..quantization.storage import scan_entries,WeightLoader,MARKER
from ..quantization.codecs import materialize


def hf_classes():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer,Qwen3MoeForCausalLM
    return Qwen3MoeConfig,Qwen3MoeDecoderLayer,Qwen3MoeForCausalLM


def inspect_qwen3_moe(path):
    root=Path(path);raw=json.loads((root/'config.json').read_text(encoding="utf-8"))
    if raw.get('model_type')!='qwen3_moe':raise ValueError('unsupported_architecture: expected qwen3_moe')
    if (root/MARKER).exists():raise ValueError('unsupported_quantization: Qwen3 MoE source only')
    for field in ('tie_word_embeddings','norm_topk_prob','output_router_logits','use_sliding_window'):
        if field in raw and type(raw[field])is not bool:raise ValueError('unsupported_capability: boolean '+field)
    if 'router_aux_loss_coef' in raw:
        value=raw['router_aux_loss_coef']
        if type(value)is not float or not math.isfinite(value) or value<0:raise ValueError('unsupported_objective: finite nonnegative float auxiliary coefficient required')
    Config,Layer,_=hf_classes();config=Config(**raw);config._attn_implementation='eager';config._experts_implementation='eager'
    if not hasattr(config,'head_dim'):
        h,n=config.hidden_size,config.num_attention_heads
        if type(h)is not int or type(n)is not int or min(h,n)<1 or h%n:raise ValueError('unsupported_capability: invalid legacy head dimensions')
        config.head_dim=h//n
    dims=[config.hidden_size,config.vocab_size,config.intermediate_size,config.moe_intermediate_size,
        config.num_hidden_layers,config.num_attention_heads,config.num_key_value_heads,config.head_dim,
        config.max_position_embeddings,config.num_experts,config.num_experts_per_tok]
    if any(type(x)is not int or x<1 for x in dims) or config.head_dim%2 or config.num_attention_heads%config.num_key_value_heads or config.num_experts_per_tok>config.num_experts:
        raise ValueError('unsupported_capability: invalid MoE/GQA dimensions')
    if config.decoder_sparse_step!=1 or config.mlp_only_layers:
        raise ValueError('unsupported_capability: only all-routed MoE layers validated')
    if config.output_router_logits:
        raise ValueError('unsupported_objective: router auxiliary output/loss not implemented; explicit CE-only source required')
    aux=config.router_aux_loss_coef
    if type(aux)not in (int,float) or not math.isfinite(aux) or aux<0:raise ValueError('unsupported_objective: invalid auxiliary coefficient')
    if config.attention_bias or getattr(config,'mlp_bias',False) or config.attention_dropout!=0 or config.hidden_act!='silu':
        raise ValueError('unsupported_capability: projection bias/dropout/activation')
    if raw.get('sliding_window')is not None or config.use_sliding_window or config.sliding_window is not None:
        raise ValueError('unsupported_capability: active/unresolved sliding window')
    types=getattr(config,'layer_types',None)
    if types is not None and types!=['full_attention']*config.num_hidden_layers:raise ValueError('unsupported_capability: layer types')
    if getattr(config,'pretraining_tp',1)!=1:raise ValueError('unsupported_capability: pretraining_tp')
    rope=config.rope_parameters
    if set(rope)-{'rope_type','rope_theta'} or rope.get('rope_type')!='default' or not math.isfinite(rope['rope_theta']) or rope['rope_theta']<=0:
        raise ValueError('unsupported_capability: default full RoPE only')
    if not math.isfinite(config.rms_norm_eps) or config.rms_norm_eps<=0:raise ValueError('unsupported_capability: norm epsilon')
    with torch.device('meta'):layer_shapes={k:list(v.shape) for k,v in Layer(config,0).state_dict().items()}
    boundaries={'model.embed_tokens.weight':[config.vocab_size,config.hidden_size],'model.norm.weight':[config.hidden_size]}
    tied=config.tie_word_embeddings is True
    if not tied:boundaries['lm_head.weight']=[config.vocab_size,config.hidden_size]
    packed=dict(boundaries);legacy=dict(boundaries)
    for i in range(config.num_hidden_layers):
        prefix=f'model.layers.{i}.'
        packed.update({prefix+k:v for k,v in layer_shapes.items()})
        legacy.update({prefix+k:v for k,v in layer_shapes.items() if k not in {'mlp.experts.gate_up_proj','mlp.experts.down_proj'}})
        for expert in range(config.num_experts):
            for name,shape in [('gate_proj',[config.moe_intermediate_size,config.hidden_size]),('up_proj',[config.moe_intermediate_size,config.hidden_size]),('down_proj',[config.hidden_size,config.moe_intermediate_size])]:
                legacy[f'{prefix}mlp.experts.{expert}.{name}.weight']=shape
    entries=scan_entries(root,'qwen3_moe')
    if set(entries)==set(legacy):layout='per_expert_2d';expected=legacy
    elif set(entries)==set(packed):layout='packed_3d';expected=packed
    else:raise ValueError('unsupported_capability: exact all-layer expert key inventory/layout required')
    if any(entries[k]['shape']!=shape for k,shape in expected.items()):raise ValueError('checkpoint/config shape mismatch')
    dtypes={v['dtype'] for v in entries.values()}
    if len(dtypes)!=1 or not dtypes<={'F32','BF16'}:raise ValueError('unsupported_capability: homogeneous FP32 or BF16 source only')
    itemsize=4 if dtypes=={'F32'} else 2
    size=lambda key:math.prod(expected[key])*itemsize
    return {'quantized_export': {'version':1,'formats':[],'rejection_reason':'Qwen3 MoE compressed path not validated'}, 'architecture':'qwen3_moe','layers':config.num_hidden_layers,'text_config':raw,'config':config,
        'entries':entries,'required':sorted(entries),'source_layout':layout,'runtime_shapes':packed,
        'source_aliases':{'lm_head.weight':'model.embed_tokens.weight'} if tied else {},
        'quantization_matrix_keys':[],'quantization_fp32_keys':[],
        'host_weight_bytes':sum(size(k) for k in entries if k not in {'lm_head.weight','model.norm.weight'}),
        'head_bytes':size('model.embed_tokens.weight' if tied else 'lm_head.weight'),
        'objective':'CE_only_native_output_router_logits_false','router_aux_loss_coef_config':aux}


def moe_feed_forward(hidden,weights,*,num_experts,top_k,norm_topk_prob):
    shape=hidden.shape;flat=hidden.view(-1,shape[-1])
    logits=F.linear(flat,weights['mlp.gate.weight'])
    probs=F.softmax(logits,dtype=torch.float32,dim=-1)
    routing,indices=torch.topk(probs,top_k,dim=-1)
    if norm_topk_prob:routing/=routing.sum(dim=-1,keepdim=True)
    routing=routing.to(logits.dtype)
    output=torch.zeros_like(flat)
    with torch.no_grad():
        mask=F.one_hot(indices,num_classes=num_experts).permute(2,1,0)
        hit=(mask.sum(dim=(-1,-2))>0).nonzero()
    for expert in hit:
        expert=expert[0];position,tokens=torch.where(mask[expert])
        current=flat[tokens]
        gate,up=F.linear(current,weights['mlp.experts.gate_up_proj'][expert]).chunk(2,dim=-1)
        value=F.linear(F.silu(gate)*up,weights['mlp.experts.down_proj'][expert])
        value=value*routing[tokens,position,None]
        output.index_add_(0,tokens,value.to(output.dtype))
    return output.reshape(shape)


class Qwen3MoeAdapter(DenseDecoderAdapter):
    family='qwen3_moe'

    def __init__(self,path,*,rank=2,alpha=4.,device='cpu',host_budget_bytes=1024**3,device_budget_bytes=2*1024**3,quantization='none',quantized_compute='bf16'):
        if quantization!='none' or quantized_compute!='bf16':raise ValueError('unsupported_quantization: Qwen3 MoE source only')
        report=inspect_qwen3_moe(path)
        if report['host_weight_bytes']>host_budget_bytes or report['head_bytes']>device_budget_bytes:raise MemoryError('resource_budget: model exceeds host/head allowance')
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('dependency: explicit CUDA unavailable')
        self.config=report['config'];self.source_aliases=report['source_aliases'];self.source_layout=report['source_layout'];self.storage_metadata=None
        loader=WeightLoader(path,report,'none');state={}
        for key in report['required']:
            record=loader.read(key);state[key]=materialize(record,self.device if key in {'lm_head.weight','model.norm.weight'} else 'cpu')
        self.identity=loader.finish();self.embedding=state.pop('model.embed_tokens.weight');self.norm=state.pop('model.norm.weight')
        self.head=self.embedding.to(self.device) if self.source_aliases else state.pop('lm_head.weight')
        assert not self.head.requires_grad and not self.embedding.requires_grad
        self.blocks=[]
        for i in range(self.config.num_hidden_layers):
            prefix=f'model.layers.{i}.';weights={k.removeprefix(prefix):state.pop(k) for k in list(state) if k.startswith(prefix)}
            if self.source_layout=='per_expert_2d':
                # Pure re-layout, same values/dtype; never re-quantize or write a base archive.
                weights['mlp.experts.gate_up_proj']=torch.stack([torch.cat([weights.pop(f'mlp.experts.{e}.gate_proj.weight'),weights.pop(f'mlp.experts.{e}.up_proj.weight')],dim=0) for e in range(self.config.num_experts)])
                weights['mlp.experts.down_proj']=torch.stack([weights.pop(f'mlp.experts.{e}.down_proj.weight') for e in range(self.config.num_experts)])
            adapters=create_lora_for_block(i,'full_attention',{k:v.shape for k,v in weights.items()},rank=rank,alpha=alpha,device=self.device,targets=ATTENTION_TARGETS['full_attention'],architecture=self.family)
            kwargs={'heads':self.config.num_attention_heads,'kv_heads':self.config.num_key_value_heads,'head_dim':self.config.head_dim,'eps':self.config.rms_norm_eps,'qk_norm':True,
                'feed_forward':functools.partial(moe_feed_forward,num_experts=self.config.num_experts,top_k=self.config.num_experts_per_tok,norm_topk_prob=self.config.norm_topk_prob),
                **{f'lora_{p}':adapters[f'self_attn.{p}_proj'] for p in ('q','v','o')}}
            self.blocks.append(BlockPlan(i,'full_attention',dense_block,kwargs,weights,adapters))
        assert not state

    def runtime_metadata(self):
        return {**super().runtime_metadata(),'source_expert_layout':self.source_layout,'runtime_expert_layout':'packed3D same dtype/value',
            'expert_transfer_granularity':'all_experts_whole_decoder_layer_synchronous','experts':self.config.num_experts,'top_k':self.config.num_experts_per_tok,
            'norm_topk_prob':self.config.norm_topk_prob,'expert_backend':'eager_per_expert_linear_index_add','objective':'CE_only','output_router_logits':False,
            'router_aux_loss_coef_config':self.config.router_aux_loss_coef,'auxiliary_loss_enabled':False,
            'objective_reason':'matches native source config output_router_logits=false; explicit enabled aux sources rejected',
            'frozen_router_and_experts_input_vjp':True,'expert_level_scheduling':False}
