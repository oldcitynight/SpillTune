"""Strict native packed gpt-oss source + synchronous whole-layer LoRA adapter.

No automatic quantizer, remote code, full expert-model instantiation or fallback.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open
from ..dense import DenseDecoderAdapter
from ..protocol import BlockPlan
from ...buffers import RawBufferContract
from ...finetune.lora import create_lora_for_block
from ...memory import pinned_layout_items
from ...quantization.storage import scan_entries,hash_tensor,MARKER
from . import ops
from .mxfp4 import validate_pair

TARGETS=['self_attn.q_proj','self_attn.v_proj','self_attn.o_proj']
RECIPE='gpt-oss-native-mxfp4-e2m1-e8m0-bf16-hf5.13-eager-v1'
RECIPE_SHA=hashlib.sha256(RECIPE.encode()).hexdigest()


def hf_classes():
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer,GptOssForCausalLM
    return GptOssConfig,GptOssDecoderLayer,GptOssForCausalLM


def source_snapshot(path,entries):
    names={Path(v['file']) for v in entries.values()}|{Path(path)/'config.json'}
    index=Path(path)/'model.safetensors.index.json'
    if index.exists():names.add(index)
    return {str(p):(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino) for p in sorted(names)}


def inspect_gpt_oss(path):
    root=Path(path);raw=json.loads((root/'config.json').read_text(encoding="utf-8"))
    if raw.get('model_type')!='gpt_oss':raise ValueError('unsupported_architecture: expected gpt_oss')
    if (root/MARKER).exists():raise ValueError('unsupported_quantization: native MXFP4 source required')
    quant=raw.get('quantization_config',{})
    if not isinstance(quant,dict) or quant.get('quant_method')!='mxfp4' or set(quant)-{'quant_method','modules_to_not_convert'}:
        raise ValueError('unsupported_quantization: exact native mxfp4 declaration required')
    if quant.get('modules_to_not_convert')!=['model.layers.*.self_attn','model.layers.*.mlp.router','model.embed_tokens','lm_head']:
        raise ValueError('unsupported_quantization: exclusions/layout not validated')
    for key,expected in [('attention_bias',True),('tie_word_embeddings',False),('output_router_logits',False),('use_cache',False)]:
        if type(raw.get(key,expected))is not bool or raw.get(key,expected)!=expected:raise ValueError('unsupported_capability: '+key)
    if raw.get('experts_per_token',raw.get('num_experts_per_tok'))!=raw.get('num_experts_per_tok'):raise ValueError('unsupported_capability: conflicting router top-k')
    Config,Layer,_=hf_classes();cfg=Config(**raw);cfg._attn_implementation='eager';cfg._experts_implementation='eager'
    dims=[cfg.hidden_size,cfg.intermediate_size,cfg.vocab_size,cfg.num_hidden_layers,cfg.num_attention_heads,cfg.num_key_value_heads,cfg.head_dim,cfg.num_local_experts,cfg.num_experts_per_tok,cfg.max_position_embeddings]
    if any(type(n)is not int or n<1 for n in dims) or cfg.hidden_size%32 or cfg.intermediate_size%32 or cfg.head_dim%2 or cfg.num_attention_heads%cfg.num_key_value_heads or cfg.num_experts_per_tok>cfg.num_local_experts:
        raise ValueError('unsupported_capability: invalid group32/expert/head dimensions')
    if cfg.attention_dropout!=0 or raw.get('hidden_act','silu')!='silu' or raw.get('swiglu_limit',7.)!=7.:
        raise ValueError('unsupported_capability: dropout/activation/limit')
    if type(cfg.sliding_window)is not int or cfg.sliding_window<1 or len(cfg.layer_types)!=cfg.num_hidden_layers or not set(cfg.layer_types)<={'full_attention','sliding_attention'}:
        raise ValueError('unsupported_capability: explicit full/sliding layer types')
    for key in ('rms_norm_eps','router_aux_loss_coef'):
        value=getattr(cfg,key)
        if type(value)not in (int,float) or not math.isfinite(value) or value<0 or (key=='rms_norm_eps' and value==0):raise ValueError('unsupported_capability: '+key)
    rope=cfg.rope_parameters;kind=rope.get('rope_type');allowed={'rope_type','rope_theta'}
    if kind=='yarn':allowed|={'factor','original_max_position_embeddings','beta_fast','beta_slow','truncate','attention_factor'}
    elif kind!='default':raise ValueError('unsupported_capability: default/YaRN RoPE only')
    if set(rope)-allowed or not math.isfinite(rope.get('rope_theta',0)) or rope['rope_theta']<=0:raise ValueError('unsupported_capability: RoPE fields')
    if kind=='yarn':
        for key in ('factor','beta_fast','beta_slow'):
            value=rope.get(key)
            if type(value)not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('unsupported_capability: YaRN '+key)
        if rope['beta_fast']<=rope['beta_slow'] or type(rope.get('original_max_position_embeddings'))is not int or rope['original_max_position_embeddings']<1 or type(rope.get('truncate',True))is not bool:
            raise ValueError('unsupported_capability: YaRN ranges')
        if 'attention_factor' in rope and (type(rope['attention_factor'])not in (int,float) or not math.isfinite(rope['attention_factor']) or rope['attention_factor']<=0):raise ValueError('unsupported_capability: YaRN attention factor')
    with torch.device('meta'):shapes={k:list(v.shape) for k,v in Layer(cfg,0).state_dict().items()}
    for name in ('gate_up_proj','down_proj'):
        e,inp,out=shapes.pop('mlp.experts.'+name)
        shapes['mlp.experts.'+name+'_blocks']=[e,out,inp//32,16]
        shapes['mlp.experts.'+name+'_scales']=[e,out,inp//32]
    expected={'model.embed_tokens.weight':[cfg.vocab_size,cfg.hidden_size],'lm_head.weight':[cfg.vocab_size,cfg.hidden_size],'model.norm.weight':[cfg.hidden_size]}
    for i in range(cfg.num_hidden_layers):expected.update({f'model.layers.{i}.'+k:v for k,v in shapes.items()})
    entries=scan_entries(root,'gpt_oss')
    if set(entries)!=set(expected) or any(entries[k]['shape']!=v for k,v in expected.items()):raise ValueError('unsupported_capability: exact native physical inventory/shape required')
    native={k for k in entries if k.endswith(('_blocks','_scales'))};dtypes={entries[k]['dtype'] for k in entries if k not in native}
    if len(dtypes)!=1 or not dtypes<={'BF16','F32'} or any(entries[k]['dtype']!='U8' for k in native):raise ValueError('unsupported_quantization: uint8 expert parts and homogeneous BF16/FP32 boundaries required')
    if dtypes=={'F32'} and (cfg.hidden_size>128 or cfg.intermediate_size>512 or cfg.num_hidden_layers>4 or cfg.num_local_experts>8 or cfg.vocab_size>4096):raise ValueError('unsupported_capability: FP32 ordinary weights validated only for explicit tiny fixtures')
    for k,v in entries.items():v['stored_bytes']=math.prod(v['shape'])*(1 if k in native else 2 if dtypes=={'BF16'} else 4)
    index=root/'model.safetensors.index.json'
    if index.exists():
        weight_map=json.loads(index.read_text(encoding="utf-8")).get('weight_map')
        if not isinstance(weight_map,dict) or set(weight_map)!=set(entries) or any(Path(entries[k]['file']).name!=v for k,v in weight_map.items()):raise ValueError('checkpoint index/physical inventory mismatch')
    size=lambda k:entries[k]['stored_bytes']
    return {'architecture':'gpt_oss','config':cfg,'text_config':raw,'entries':entries,'required':sorted(entries),'source_aliases':{},
        'layers':cfg.num_hidden_layers,'host_weight_bytes':sum(size(k) for k in entries if k not in {'model.norm.weight','lm_head.weight'}),'head_bytes':size('lm_head.weight'),
        'quantization_matrix_keys':[],'quantization_fp32_keys':[],
        'quantized_export':{'version':1,'formats':[],'rejection_reason':'native MXFP4 re-encoding/export is not validated'}}


class NativeLoader:
    def __init__(self,path,report,device,host_budget_bytes):
        self.path=Path(path);self.report=report;self.device=torch.device(device);self.before=source_snapshot(path,report['entries'])
        self.parts={};self.owners={};self.capacities=[];self.read_keys=[]
        self.digest=hashlib.sha256(json.dumps(report['text_config'],sort_keys=True).encode());self.max_source_tensor_bytes=0
        if self.device.type=='cuda':
            layout=pinned_layout_items([(v['stored_bytes'],k,'value') for k,v in report['entries'].items() if k.startswith('model.layers.')],512*1024**2)
            self.capacities=[a['capacity'] for a in layout]
            for i,a in enumerate(layout):
                for k,_,offset,size in a['items']:self.parts[k]=(i,offset,size)
        required_host=max(report['host_weight_bytes'],sum(self.capacities)+report['entries']['model.embed_tokens.weight']['stored_bytes'])
        if required_host>host_budget_bytes:raise MemoryError('resource_budget: actual pinned capacity plus CPU embedding exceeds host allowance before payload')

    def read(self,key):
        if key in self.read_keys:raise ValueError('native_mxfp4: repeated source payload read')
        entry=self.report['entries'][key]
        with safe_open(entry['file'],framework='pt',device='cpu') as handle:value=handle.get_tensor(key).detach()
        hash_tensor(self.digest,key,value);self.read_keys.append(key);self.max_source_tensor_bytes=max(self.max_source_tensor_bytes,entry['stored_bytes'])
        if value.dtype!=torch.uint8:
            view=value.reshape(-1)
            for start in range(0,view.numel(),1024**2):
                if not bool(torch.isfinite(view[start:start+1024**2]).all()):raise ValueError('source contains non-finite ordinary tensor')
        planned=self.parts.get(key)
        if planned is not None:
            index,offset,size=planned
            if index not in self.owners:self.owners[index]=torch.empty(self.capacities[index],dtype=torch.uint8,pin_memory=True)
            dest=self.owners[index].narrow(0,offset,size).view(value.dtype).view(value.shape);dest.copy_(value);value=dest
        elif key in {'lm_head.weight','model.norm.weight'}:value=value.to(self.device)
        return value

    def finish(self,state):
        if self.read_keys!=self.report['required']:raise ValueError('native source loading inventory/order mismatch')
        for k,v in state.items():
            if k.endswith('_blocks'):validate_pair(v,state[k.removesuffix('_blocks')+'_scales'])
        if source_snapshot(self.path,self.report['entries'])!=self.before or scan_entries(self.path,'gpt_oss')!={k:{a:b for a,b in v.items() if a!='stored_bytes'} for k,v in self.report['entries'].items()}:
            raise ValueError('source mutated while loading')
        self.source_identity=self.digest.hexdigest()
        return hashlib.sha256((self.source_identity+RECIPE_SHA).encode()).hexdigest()


def working_set(report,rank):
    """Source-metadata-only peak reservation, before any payload read."""
    cfg=report['config'];size=4 if report['entries']['lm_head.weight']['dtype']=='F32' else 2
    expert_bytes=cfg.num_local_experts*3*cfg.hidden_size*cfg.intermediate_size*size
    extra_matrix=cfg.hidden_size*2*cfg.intermediate_size*max(size,2)
    workspace=max(2*1024**3 if cfg.hidden_size>=1024 else 0,expert_bytes+extra_matrix+32*1024**2)
    slots=[]
    for i in range(cfg.num_hidden_layers):
        offset=0;prefix=f'model.layers.{i}.'
        for key in report['required']:
            if key.startswith(prefix):offset=(offset+255)//256*256+report['entries'][key]['stored_bytes']
        slots.append(offset)
    trainables=0
    for i in range(cfg.num_hidden_layers):
        for target in TARGETS:
            out,inp=report['entries'][f'model.layers.{i}.{target}.weight']['shape'];trainables+=rank*(out+inp)
    state=trainables*4*8
    reserve=(512 if cfg.hidden_size>=1024 else 32)*1024**2
    boundaries=report['head_bytes']+report['entries']['model.norm.weight']['stored_bytes']
    return {'workspace':workspace,'slot':max(slots),'lora_state':state,'execution_reserve':reserve,'device_charge':boundaries+max(slots)+255+workspace+state+reserve}


class GptOssAdapter(DenseDecoderAdapter):
    family='gpt_oss'

    def __init__(self,path,*,rank=2,alpha=4.,device='cpu',host_budget_bytes=1024**3,device_budget_bytes=2*1024**3,quantization='none',quantized_compute='bf16'):
        if quantization!='none' or quantized_compute!='bf16':raise ValueError('unsupported_quantization: native MXFP4 storage and BF16 dequantization only')
        report=inspect_gpt_oss(path)
        if type(rank)is not int or rank<1:raise ValueError('rank must be positive integer')
        if type(alpha)not in (int,float) or not math.isfinite(alpha) or alpha<0:raise ValueError('alpha must be finite nonnegative')
        budget=working_set(report,rank)
        if budget['device_charge']>device_budget_bytes:raise MemoryError('resource_budget: head, slot, decode workspace, LoRA state and execution reserve exceed device allowance before payload')
        if torch.device(device).type=='cpu' and report['host_weight_bytes']+budget['device_charge']>host_budget_bytes:raise MemoryError('resource_budget: CPU source plus complete working set exceeds host allowance before payload')
        self.device=torch.device(device);self.config=report['config'];self.source_aliases={};self.storage_metadata=None
        loader=NativeLoader(path,report,self.device,host_budget_bytes)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('dependency: explicit CUDA unavailable')
        state={key:loader.read(key) for key in report['required']};self.identity=loader.finish(state)
        self.source_identity=loader.source_identity;self.source_read_keys=list(loader.read_keys);self.pinned_owners=loader.owners;self.pinned_capacities=loader.capacities
        self.embedding=state.pop('model.embed_tokens.weight');self.head=state.pop('lm_head.weight');self.norm=state.pop('model.norm.weight')
        self.blocks=[];cfg=self.config
        workspace=budget['workspace'];self.working_set_budget=budget
        contract=RawBufferContract(RECIPE,RECIPE_SHA,workspace)
        for i,kind in enumerate(cfg.layer_types):
            prefix=f'model.layers.{i}.';weights={k.removeprefix(prefix):state.pop(k) for k in list(state) if k.startswith(prefix)}
            adapters=create_lora_for_block(i,kind,{k:v.shape for k,v in weights.items()},rank=rank,alpha=alpha,device=self.device,targets=TARGETS,architecture=self.family)
            kwargs={'heads':cfg.num_attention_heads,'kv_heads':cfg.num_key_value_heads,'head_dim':cfg.head_dim,'eps':cfg.rms_norm_eps,'experts':cfg.num_local_experts,'top_k':cfg.num_experts_per_tok,
                **{f'lora_{p}':adapters[f'self_attn.{p}_proj'] for p in ('q','v','o')}}
            self.blocks.append(BlockPlan(i,kind,ops.block,kwargs,weights,adapters,buffer_contract=contract))
        assert not state

    def positions_and_masks(self,valid):
        if valid.dtype!=torch.bool or valid.ndim!=2 or valid.shape[1]<1 or valid.shape[1]>self.config.max_position_embeddings or not bool(valid[:,0].all()) or bool(((~valid[:,:-1])&valid[:,1:]).any()):
            raise ValueError('unsupported_capability: all-valid/right-padding within context required')
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
        valid=valid.to(self.device);b,n=valid.shape;positions=torch.arange(n,device=self.device).expand(b,-1)
        # The fixed, local HF RoPE primitive; no model construction or downloads.
        rotary=GptOssRotaryEmbedding(self.config,device=self.device)(torch.empty((),device=self.device,dtype=self.head.dtype),positions)
        q=torch.arange(n,device=self.device)[:,None];k=torch.arange(n,device=self.device)[None,:]
        masks={}
        for kind in set(self.config.layer_types):
            blocked=(k>q) | ((k<=q-self.config.sliding_window) if kind=='sliding_attention' else False)
            blocked=blocked[None,None] | ~valid[:,None,None,:]
            masks[kind]=torch.zeros(blocked.shape,device=self.device,dtype=self.head.dtype).masked_fill(blocked,torch.finfo(self.head.dtype).min)
        return rotary,masks

    def terminal_norm(self,hidden):return ops.rms(hidden,self.norm,self.config.rms_norm_eps)

    def runtime_metadata(self):
        return {**super().runtime_metadata(),'rope':self.config.rope_parameters['rope_type'],'attention':'HF5.13 eager GQA with per-head sink','source_storage':'native_mxfp4_e2m1_e8m0','compute':'BF16 dequantized experts (FP32 explicit tiny ordinary fixture supported)',
            'native_lowbit_gemm':False,'decode_recipe':RECIPE_SHA,'source_identity':self.source_identity,'source_read_count':len(self.source_read_keys),'direct_pinned_capacity_bytes':sum(self.pinned_capacities),
            'expert_transfer_granularity':'all_experts_whole_layer_synchronous','expert_decode':'per_hit_expert_per_functional; graph-retained matrices charged to workspace','expert_decode_trace_limitation':'generic raw trace bytes/time describe staging, not adapter expert decode work',
            'objective':'CE_only','output_router_logits':False,'router_aux_loss_coef_config':self.config.router_aux_loss_coef,'auxiliary_loss_enabled':False,'frozen_router_and_experts_input_vjp':True,'targets':TARGETS,
            'input_scope':'token IDs only; Harmony/chat/tool semantics not accepted','source_quantized_base_required':True,'metadata_working_set_budget':self.working_set_budget}

    def close(self):
        super().close();self.pinned_owners.clear()
