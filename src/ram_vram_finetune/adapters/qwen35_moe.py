"""Native Qwen3.5-MoE text CE-LoRA with a frozen synchronous whole-layer slot.

Qwen3.6 sources declaring this architecture are inspected without aliases.
Native Transformers eager decoder graphs are stateless templates: their meta
parameters are borrowed from the slot only inside functional_call. LoRA is
injected at real projections; frozen routers, all experts and recurrent token
mixers remain differentiable with respect to the block input.
"""
import json,math,hashlib,struct
from pathlib import Path
import torch
from .qwen35.adapter import Qwen35Adapter
from .qwen35.blocks import _linear_with_lora
from .protocol import BlockPlan
from .targets import ATTENTION_TARGETS
from ..finetune.lora import create_lora_for_block
from ..quantization.storage import scan_entries,WeightLoader,estimated_bytes,MARKER
from ..quantization.codecs import materialize,flatten_records


def hf_classes():
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as h
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    return Qwen3_5MoeTextConfig,h


def source_snapshot(path):
    root=Path(path);out={}
    for p in sorted(root.glob('*.safetensors')):
        with p.open('rb') as f:
            prefix=f.read(8);n=struct.unpack('<Q',prefix)[0]
            if n>64*1024**2:raise ValueError('unsupported_capability: excessive source header')
            data=f.read(n)
        if len(data)!=n:raise ValueError('source header truncated')
        st=p.stat();out[p.name]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'header_sha256':hashlib.sha256(prefix+data).hexdigest()}
    for name in ['config.json','model.safetensors.index.json']:
        p=root/name
        if p.exists():out[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return out


def inspect_qwen35_moe(path):
    root=Path(path);raw=json.loads((root/'config.json').read_text(encoding="utf-8"));text=raw.get('text_config',raw)
    if raw.get('model_type') not in {'qwen3_5_moe','qwen3_5_moe_text'} or text.get('model_type')!='qwen3_5_moe_text':raise ValueError('unsupported_architecture: native qwen3_5_moe text required')
    if any('quantization_config' in c for c in (raw,text)):raise ValueError('unsupported_quantization: external compressed source configuration')
    if (root/MARKER).exists():raise ValueError('unsupported_quantization: source-only Qwen3.5-MoE; base export not validated')
    for c in (raw,text):
        for k in ['tie_word_embeddings','attention_bias','output_router_logits','attn_output_gate','mtp_use_dedicated_embeddings']:
            if k in c and type(c[k]) is not bool:raise ValueError('unsupported_capability: boolean '+k)
    if raw.get('tie_word_embeddings',False) or text.get('tie_word_embeddings',False):raise ValueError('unsupported_capability: tied Qwen3.5-MoE not validated')
    if text.get('output_router_logits',False):raise ValueError('unsupported_objective: CE-only; auxiliary training not implemented')
    aux=text.get('router_aux_loss_coef',.001)
    if type(aux) not in (int,float) or not math.isfinite(aux) or aux<0:raise ValueError('unsupported_objective: invalid auxiliary coefficient')
    if 'num_hidden_layers' in text and (type(text['num_hidden_layers']) is not int or text['num_hidden_layers']<1):raise ValueError('unsupported_capability: positive layer count')
    for key in ['hidden_size','vocab_size','num_attention_heads','num_key_value_heads','head_dim','num_experts','num_experts_per_tok','moe_intermediate_size','shared_expert_intermediate_size','linear_num_value_heads','linear_num_key_heads','linear_key_head_dim','linear_value_head_dim','linear_conv_kernel_dim','max_position_embeddings']:
        if key in text and (type(text[key]) is not int or text[key]<1):raise ValueError('unsupported_capability: positive integer '+key)
    Config,h=hf_classes();cfg=Config(**text);cfg._attn_implementation='eager';cfg._experts_implementation='eager'
    dims=['hidden_size','vocab_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','num_experts','num_experts_per_tok','moe_intermediate_size','shared_expert_intermediate_size','linear_num_value_heads','linear_num_key_heads','linear_key_head_dim','linear_value_head_dim','linear_conv_kernel_dim','max_position_embeddings']
    if any(type(getattr(cfg,k)) is not int or getattr(cfg,k)<1 for k in dims):raise ValueError('unsupported_capability: positive integer dimensions')
    if cfg.num_experts_per_tok>cfg.num_experts or cfg.num_attention_heads%cfg.num_key_value_heads or cfg.linear_num_value_heads%cfg.linear_num_key_heads:raise ValueError('unsupported_capability: head/expert dimensions')
    if cfg.attention_dropout!=0 or cfg.attention_bias or cfg.hidden_act!='silu' or cfg.rms_norm_eps!=1e-6 or text.get('attn_output_gate',True) is not True:raise ValueError('unsupported_capability: bias/dropout/activation/norm/gate')
    if len(cfg.layer_types)!=cfg.num_hidden_layers or set(cfg.layer_types)-{'linear_attention','full_attention'}:raise ValueError('unsupported_capability: exact hybrid schedule required')
    rope=cfg.rope_parameters;allowed={'rope_type','rope_theta','partial_rotary_factor','mrope_interleaved','mrope_section'}
    dim=cfg.head_dim*rope.get('partial_rotary_factor',1.)
    if set(rope)-allowed or rope.get('rope_type')!='default' or not math.isfinite(rope['rope_theta']) or rope['rope_theta']<=0 or not dim.is_integer() or int(dim)%2 or not 0<dim<=cfg.head_dim or rope.get('mrope_interleaved',True) is not True:raise ValueError('unsupported_capability: default partial interleaved text RoPE only')
    section=rope.get('mrope_section',[11,11,10])
    if not isinstance(section,list) or len(section)!=3 or any(type(x) is not int or x<0 for x in section) or sum(section)!=int(dim)//2:raise ValueError('unsupported_capability: mrope section')
    entries=scan_entries(root,'qwen35_moe');prefix='model.language_model.';expected={prefix+'embed_tokens.weight':[cfg.vocab_size,cfg.hidden_size],prefix+'norm.weight':[cfg.hidden_size],'lm_head.weight':[cfg.vocab_size,cfg.hidden_size]};shapes={}
    for i,kind in enumerate(cfg.layer_types):
        if kind not in shapes:
            with torch.device('meta'):layer=h.Qwen3_5MoeDecoderLayer(cfg,i)
            shapes[kind]={k:list(v.shape) for k,v in layer.state_dict().items()}
        expected.update({prefix+f'layers.{i}.'+k:v for k,v in shapes[kind].items()})
    excluded={k:v for k,v in entries.items() if k not in expected}
    vision={k:v for k,v in excluded.items() if k.startswith('model.visual.')};mtp={k:v for k,v in excluded.items() if k.startswith('mtp.')}
    if set(excluded)!=set(vision)|set(mtp):raise ValueError('unsupported_capability: unknown extra source tensors')
    if ('vision_config' in raw)!=bool(vision):raise ValueError('unsupported_capability: declared vision branch must match source inventory')
    if 'mtp_num_hidden_layers' in text and (type(text['mtp_num_hidden_layers']) is not int or text['mtp_num_hidden_layers'] not in (0,1)):raise ValueError('unsupported_capability: explicit zero or one MTP layer')
    if bool(text.get('mtp_num_hidden_layers',0))!=bool(mtp):raise ValueError('unsupported_capability: declared MTP branch must match source inventory')
    if vision:
        if 'vision_config' not in raw:raise ValueError('unsupported_capability: unbound vision configuration')
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
        with torch.device('meta'):visual=h.Qwen3_5MoeVisionModel(Qwen3_5MoeVisionConfig(**raw['vision_config']))
        visexpected={'model.visual.'+k:list(v.shape) for k,v in visual.state_dict().items()}
        if {k:v['shape'] for k,v in vision.items()}!=visexpected:raise ValueError('unsupported_capability: exact excluded vision schema')
    if mtp:
        if text.get('mtp_num_hidden_layers')!=1 or text.get('mtp_use_dedicated_embeddings',False):raise ValueError('unsupported_capability: excluded MTP recipe')
        if 'full_attention' not in shapes:raise ValueError('unsupported_capability: MTP full decoder shape absent')
        mex={'mtp.layers.0.'+k:v for k,v in shapes['full_attention'].items()}
        mex.update({'mtp.fc.weight':[cfg.hidden_size,2*cfg.hidden_size],**{'mtp.'+k+'.weight':[cfg.hidden_size] for k in ['norm','pre_fc_norm_embedding','pre_fc_norm_hidden']}})
        if {k:v['shape'] for k,v in mtp.items()}!=mex:raise ValueError('unsupported_capability: exact excluded MTP schema')
    if not set(expected)<=set(entries) or any(entries[k]['shape']!=v for k,v in expected.items()):raise ValueError('checkpoint/config exact text shape mismatch')
    if any(v['dtype']!='BF16' for v in entries.values()):raise ValueError('unsupported_capability: homogeneous BF16 source only')
    index=root/'model.safetensors.index.json'
    if index.exists() and json.loads(index.read_text(encoding="utf-8")).get('weight_map')!={k:Path(v['file']).name for k,v in entries.items()}:raise ValueError('unsupported_capability: exact source shard index')
    matrices=[k for k in expected if k.startswith(prefix+'layers.') and len(entries[k]['shape']) in (2,3) and math.prod(entries[k]['shape'])>=128 and not k.endswith(('.mlp.gate.weight','.mlp.shared_expert_gate.weight','linear_attn.conv1d.weight'))]
    fp32=[k for k in expected if k.endswith(('.A_log','.dt_bias'))]
    return {'source_snapshot':source_snapshot(root),'architecture':'qwen35_moe','config':cfg,'layers':cfg.num_hidden_layers,'layer_types':cfg.layer_types,'entries':entries,'required':sorted(entries),'active_required':sorted(expected),'excluded_required':sorted(excluded),'text_config':{'text':text,'excluded_vision_config':raw.get('vision_config'),'recipe':'text CE only, excluded vision/MTP raw hash bound v1'},'source_aliases':{},'training_scope':'text causal cross-entropy only; vision/MTP/aux excluded','excluded_nontext_tensors':{'vision':sorted(vision),'mtp':sorted(mtp)},'quantization_matrix_keys':matrices,'quantization_fp32_keys':fp32,'quantized_export':{'version':1,'formats':[],'rejection_reason':'Qwen3.5-MoE compressed base archive export not validated'},'host_weight_bytes':sum(math.prod(entries[k]['shape'])*2 for k in expected if k not in {'lm_head.weight',prefix+'norm.weight'}),'head_bytes':cfg.vocab_size*cfg.hidden_size*2}


class _Projection(torch.nn.Module):
    def __init__(self,weight,projection):
        super().__init__();self.weight=weight;self.projection=projection
    def forward(self,x):return _linear_with_lora(x,self.weight,None,self.projection)


def template(cfg,index,adapters):
    _,h=hf_classes()
    with torch.device('meta'):layer=h.Qwen3_5MoeDecoderLayer(cfg,index)
    layer.eval().requires_grad_(False)
    if cfg.layer_types[index]=='linear_attention':
        core=layer.linear_attn
        # Explicit backend selection, never optional fused-kernel fallback.
        core.causal_conv1d_fn=None;core.causal_conv1d_update=h.torch_causal_conv1d_update
        core.chunk_gated_delta_rule=h.torch_chunk_gated_delta_rule;core.recurrent_gated_delta_rule=h.torch_recurrent_gated_delta_rule
        with torch.device('meta'):core.norm=h.Qwen3_5MoeRMSNormGated(cfg.linear_value_head_dim,eps=cfg.rms_norm_eps)
    for target,p in adapters.items():
        parent,leaf=target.rsplit('.',1);owner=layer.get_submodule(parent);base=getattr(owner,leaf)
        if not isinstance(base,torch.nn.Linear) or base.bias is not None:raise ValueError('unsupported_capability: real bias-free LoRA projection required')
        setattr(owner,leaf,_Projection(base.weight,p))
    return layer


def native_block(hidden_states,position_embeddings,attention_mask,*,native_template,**kwargs):
    from ..finetune.lora import LoRAProjection
    weights={k:v for k,v in kwargs.items() if not isinstance(v,LoRAProjection)}
    if any(p.device.type!='meta' for p in native_template.parameters()):raise RuntimeError('native template retained slot weights')
    result=torch.func.functional_call(native_template,weights,(hidden_states,),{'position_embeddings':position_embeddings,'attention_mask':attention_mask,'past_key_values':None},strict=True)
    if any(p.device.type!='meta' for p in native_template.parameters()):raise RuntimeError('native template failed to release slot weights')
    return result


class Qwen35MoeAdapter(Qwen35Adapter):
    def __init__(self,path,*,rank=2,alpha=4.,device='cpu',host_budget_bytes=1024**3,device_budget_bytes=2*1024**3,quantization='none',quantized_compute='bf16'):
        if type(rank) is not int or rank<1 or type(alpha) not in (int,float) or not math.isfinite(alpha) or alpha<0:raise ValueError('invalid rank/alpha')
        if quantization not in {'none','int4-sym-g128'} or quantized_compute!='bf16':raise ValueError('unsupported_quantization: Qwen3.5-MoE supports BF16 source or explicit INT4 storage/BF16 compute')
        report=inspect_qwen35_moe(path);self.config=report['config'];self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('dependency: explicit CUDA unavailable')
        prefix='model.language_model.';boundary={'lm_head.weight',prefix+'norm.weight'};decoder=[k for k in report['active_required'] if k.startswith(prefix+'layers.')]
        from ..memory import pinned_layout_items
        items=[]
        for k in decoder:
            count=math.prod(report['entries'][k]['shape']);parts={'payload':((count+127)//128)*64,'scales':((count+127)//128)*4} if quantization!='none' and k in report['quantization_matrix_keys'] else {'value':estimated_bytes(report,quantization,k)}
            items.extend((n,k,p) for p,n in parts.items())
        pinned=sum(a['capacity'] for a in pinned_layout_items(items,512*1024**2));embedding=estimated_bytes(report,quantization,prefix+'embed_tokens.weight')
        head=estimated_bytes(report,quantization,'lm_head.weight')
        if pinned+embedding>host_budget_bytes or head>device_budget_bytes:raise MemoryError('resource_budget: pinned decoder plus CPU embedding/head allowance before payload')
        decoded=[];uploads=[]
        for i in range(self.config.num_hidden_layers):
            keys=[k for k in decoder if k.startswith(prefix+f'layers.{i}.')];offset=0;upload=0
            for k in sorted(keys):
                count=math.prod(report['entries'][k]['shape']);offset=(offset+255)//256*256+count*(4 if k in report['quantization_fp32_keys'] else 2)
                if quantization!='none' and k in report['quantization_matrix_keys']:
                    for n in [((count+127)//128)*64,((count+127)//128)*4]:upload=(upload+255)//256*256+n
                else:upload=(upload+255)//256*256+estimated_bytes(report,quantization,k)
            decoded.append(offset);uploads.append(upload)
        # Reserve includes all eight FP32 LoRA fields and bounded decode/terminal scratch.
        targets=sum(1 if k=='linear_attention' else 3 for k in self.config.layer_types)
        lora_upper=targets*rank*4*self.config.hidden_size*8*4*4
        combined=head+self.config.hidden_size*2+max(decoded)+max(uploads)+lora_upper+256*1024**2
        if combined>device_budget_bytes:raise MemoryError('resource_budget: combined boundary/slot/upload/LoRA/scratch allowance before payload')
        max_source=max(math.prod(v['shape'])*2 for v in report['entries'].values())
        max_decoded=max(math.prod(report['entries'][k]['shape'])*(4 if k in report['quantization_fp32_keys'] else 2) for k in decoder)
        max_encoded=max(estimated_bytes(report,quantization,k) for k in decoder)
        host_combined=pinned+embedding+max_source+max_encoded+max_decoded+min(2*1024**3,pinned)
        if host_combined>host_budget_bytes:raise MemoryError('resource_budget: host includes source/encoder/native-decode tensor and model-proportional process/scratch reserve capped at 2GiB')
        self.prepayload_budget={'pinned_capacity_bytes':pinned,'cpu_embedding_bytes':embedding,'largest_source_tensor_bytes':max_source,'largest_encoded_tensor_bytes':max_encoded,'largest_native_decode_cpu_tensor_bytes':max_decoded,'host_combined_bytes':host_combined,'production_device_combined_bytes':combined,'local_native_device_combined_bytes':combined+max(decoded),'local_native_decoder_bytes':max(decoded)}
        if source_snapshot(path)!=report['source_snapshot']:raise ValueError('source changed before payload loading')
        direct=quantization=='int4-sym-g128' and self.device.type=='cuda'
        # Excluded source is raw-hashed but immediately discarded; no pin/quantize.
        loader=WeightLoader(path,report,quantization,pin_keys=decoder if direct else (),host_budget_bytes=host_budget_bytes,pinned_arena_limit=512*1024**2)
        state={}
        for k in report['required']:
            if k in report['excluded_required']:
                from safetensors import safe_open
                from ..quantization.storage import hash_tensor
                with safe_open(report['entries'][k]['file'],framework='pt',device='cpu') as handle:value=handle.get_tensor(k)
                from ..quantization.codecs import finite
                if not finite(value):raise ValueError('nonfinite excluded source weight')
                hash_tensor(loader.original,k,value);loader.stored.update(('excluded:'+k).encode());loader.read_keys.append(k);del value
            else:
                record=loader.read(k);state[k]=materialize(record,self.device) if k in boundary else record
        if source_snapshot(path)!=report['source_snapshot']:raise ValueError('source changed during payload loading')
        self.identity=loader.finish();self.source_read_keys=list(loader.read_keys);self.source_aliases={};self.training_scope=report['training_scope'];self.excluded_nontext_tensors=report['excluded_nontext_tensors']
        self.embedding=materialize(state.pop(prefix+'embed_tokens.weight'));self.norm=state.pop(prefix+'norm.weight');self.head=state.pop('lm_head.weight')
        _,h=hf_classes();self.rotary=h.Qwen3_5MoeTextRotaryEmbedding(self.config,device=self.device).eval();self.blocks=[];self.cores=[]
        for i,kind in enumerate(report['layer_types']):
            pre=prefix+f'layers.{i}.';records={k.removeprefix(pre):state.pop(k) for k in list(state) if k.startswith(pre)}
            adapters=create_lora_for_block(i,kind,{k:v.shape for k,v in records.items()},rank=rank,alpha=alpha,device=self.device,targets=ATTENTION_TARGETS[kind],architecture='Qwen3.5-MoE')
            native=template(self.config,i,adapters);tensors,specs=flatten_records(records)
            self.blocks.append(BlockPlan(i,kind,native_block,{'native_template':native,**{'lora_'+str(n):p for n,p in enumerate(adapters.values())}},tensors,adapters,specs if quantization!='none' else None))
        assert not state
        self.loading_metadata={'format':quantization,'compute':'bf16' if quantization!='none' else 'source','source_identity':loader.source_identity,'storage_identity':loader.storage_identity,'total_stored_bytes':loader.stored_bytes,'max_source_tensor_bytes':loader.max_source_tensor_bytes,'quantized_matrices':loader.quantized_tensors,'full_bf16_decoder_retained':False,'direct_pinned_capacity_bytes':sum(loader.pinned_capacities),'decoder_stored_bytes':sum(t.numel()*t.element_size() for b in self.blocks for t in b.host_weights.values()),'cpu_embedding_bytes':embedding,'device_boundary_bytes':head+self.norm.numel()*self.norm.element_size(),'quantized_scope':'decoder 2D/3D matrices >=128; router/shared-gate, boundaries, norms/conv excluded','host_loading':'direct_packed_pinned' if direct else 'pageable','excluded_source_raw_hash_bound':True}
        self.storage_metadata=self.loading_metadata if quantization!='none' else None
    def runtime_metadata(self):
        return {'architecture':'qwen3_5_moe_text','layer_count':len(self.blocks),'linear_backends':['explicit Transformers torch_chunk_gated_delta_rule and torch conv1d'],'expert_backend':'native eager all-expert whole-layer slot','objective':'text_CE_only','auxiliary_loss_enabled':False,'source_aliases':{},'training_scope':self.training_scope,'excluded_nontext_tensor_counts':{k:len(v) for k,v in self.excluded_nontext_tensors.items()},'asynchronous_transfers':'unsupported','meta_template_parameters_resident_bytes':0,'prepayload_budget':self.prepayload_budget,'storage_compute':self.loading_metadata}
