"""Strict native GLM-4 MoE Lite CE backbone; explicit MTP exclusion.

Reference: installed Transformers5.13 GLM4MoeLite, Apache-2.0.
Copyright 2025 the HuggingFace Team. All rights reserved.
See LICENSES/transformers-APACHE-2.0.txt and legal/REFERENCE_NOTICES.md.
MLA/grouped MoE equations share the already isolated DeepSeek functional kernel;
configuration, layout, storage/transfer and objective policy are GLM-owned.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open
from .deepseek import DeepseekV3Adapter,deepseek_block
from .protocol import BlockPlan
from ..finetune.lora import create_lora_for_block
from ..memory import pinned_layout_items
from ..quantization.storage import WeightLoader,scan_entries,estimated_bytes,hash_tensor,MARKER
from ..quantization.codecs import flatten_records,materialize,finite,FORMATS,stride_for

RECIPE='glm4-moe-lite-ce-only-exact-mtp-hash-discard-g128-bf16-v1'
RECIPE_SHA=hashlib.sha256(RECIPE.encode()).hexdigest()
TARGETS=['self_attn.q_a_proj','self_attn.q_b_proj','self_attn.kv_a_proj_with_mqa','self_attn.kv_b_proj','self_attn.o_proj']
CAPABILITY={'version':1,'formats':[],'rejection_reason':'GLM CE-only active subset; full native MTP base archive export not validated'}


def hf_classes():
    from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer,Glm4MoeLiteForCausalLM
    return Glm4MoeLiteConfig,Glm4MoeLiteDecoderLayer,Glm4MoeLiteForCausalLM


def source_snapshot(path,entries):
    files={Path(v['file']) for v in entries.values()}|{Path(path)/'config.json'}
    if (Path(path)/'model.safetensors.index.json').exists():files.add(Path(path)/'model.safetensors.index.json')
    return {str(p):(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino) for p in sorted(files)}


def inspect_glm4_moe_lite(path):
    root=Path(path);raw=json.loads((root/'config.json').read_text(encoding="utf-8"))
    def reject(reason):raise ValueError('unsupported_glm4_moe_lite: '+reason)
    if raw.get('model_type')!='glm4_moe_lite':reject('native glm4_moe_lite only; no architecture aliases')
    if raw.get('architectures') not in (None,['Glm4MoeLiteForCausalLM']):reject('causal LM architecture required')
    if (root/MARKER).exists() or raw.get('quantization_config') is not None:reject('explicit native BF16/F32 source required')
    for k in ['aux_loss_alpha','router_aux_loss_coef','num_nextn_layers','num_mtp_layers','mtp_num_hidden_layers']:
        if raw.get(k,0) not in (0,None):reject('unsupported auxiliary/MTP declaration '+k)
    mtp=raw.get('num_nextn_predict_layers',0)
    if type(mtp)is not int or mtp not in (0,1):reject('only explicit zero/one excluded native MTP branch')
    if raw.get('output_router_logits',False) is not False:reject('CE-only no auxiliary router objective')
    if raw.get('ep_size',1)!=1 or raw.get('pretraining_tp',1)!=1:reject('expert/tensor parallel')
    if raw.get('moe_layer_freq',1)!=1 or raw.get('first_k_dense_replace',1)!=1:reject('only firstdense then contiguous sparse schedule')
    if raw.get('scoring_func','sigmoid')!='sigmoid' or raw.get('topk_method','noaux_tc')!='noaux_tc':reject('sigmoid/noaux_tc router required')
    if raw.get('sliding_window') is not None or raw.get('use_sliding_window',False):reject('sliding window')
    if any(k.startswith(('index_','hc_','compress_')) for k in raw):reject('unsupported attention variant')
    if any(type(raw.get(k,True))is not bool for k in ['norm_topk_prob','rope_interleave']):reject('router/rotary boolean fields')
    Config,Layer,_=hf_classes();cfg=Config(**raw);cfg._attn_implementation='eager';cfg._experts_implementation='eager';cfg.use_cache=False
    dims=['hidden_size','vocab_size','intermediate_size','moe_intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','kv_lora_rank','q_lora_rank','qk_nope_head_dim','qk_rope_head_dim','v_head_dim','n_routed_experts','n_shared_experts','n_group','topk_group','num_experts_per_tok','max_position_embeddings']
    if any(type(getattr(cfg,k))is not int or getattr(cfg,k)<1 for k in dims):reject('positive dimensions and low-rank q branch required')
    if cfg.num_hidden_layers<2 or cfg.mlp_layer_types!=['dense']+['sparse']*(cfg.num_hidden_layers-1):reject('native firstdense/sparse layer schedule required')
    if cfg.tie_word_embeddings or cfg.attention_bias or raw.get('mlp_bias',False) or cfg.hidden_act!='silu' or cfg.attention_dropout!=0:reject('shared embeddings/bias/activation/dropout')
    if cfg.num_key_value_heads!=cfg.num_attention_heads or cfg.qk_rope_head_dim%2:reject('MLA head grouping/rotary dimensions')
    if type(cfg.norm_topk_prob)is not bool or type(cfg.rope_interleave)is not bool:reject('router/rotary booleans')
    if cfg.n_routed_experts%cfg.n_group or cfg.n_routed_experts//cfg.n_group<2 or cfg.topk_group>cfg.n_group or cfg.num_experts_per_tok>cfg.topk_group*(cfg.n_routed_experts//cfg.n_group):reject('group/top-k capacity')
    if not all(type(x)in (int,float) and math.isfinite(x) and x>0 for x in [cfg.rms_norm_eps,cfg.routed_scaling_factor]):reject('epsilon/router scale')
    rope=cfg.rope_parameters
    if set(rope)-{'rope_type','rope_theta','partial_rotary_factor'} or rope.get('rope_type')!='default' or not math.isfinite(rope.get('rope_theta',0)) or rope['rope_theta']<=1 or rope.get('partial_rotary_factor',1.)!=1.:reject('only validated default RoPE')
    if raw.get('partial_rotary_factor',1.)!=1.:reject('partial MLA rotary override')
    entries=scan_entries(root,'glm4_moe_lite');expected={}
    with torch.device('meta'):
        for i in range(cfg.num_hidden_layers):expected.update({f'model.layers.{i}.'+k:list(v.shape) for k,v in Layer(cfg,i).state_dict().items()})
    packed='model.layers.1.mlp.experts.gate_up_proj' in entries
    if not packed:
        for i in range(1,cfg.num_hidden_layers):
            prefix=f'model.layers.{i}.mlp.experts.';expected.pop(prefix+'gate_up_proj');expected.pop(prefix+'down_proj')
            for e in range(cfg.n_routed_experts):
                for name,shape in [('gate_proj',[cfg.moe_intermediate_size,cfg.hidden_size]),('up_proj',[cfg.moe_intermediate_size,cfg.hidden_size]),('down_proj',[cfg.hidden_size,cfg.moe_intermediate_size])]:expected[prefix+f'{e}.{name}.weight']=shape
    expected.update({'model.embed_tokens.weight':[cfg.vocab_size,cfg.hidden_size],'model.norm.weight':[cfg.hidden_size],'lm_head.weight':[cfg.vocab_size,cfg.hidden_size]})
    active=set(expected);excluded=set()
    if mtp:
        prefix=f'model.layers.{cfg.num_hidden_layers}.';last=f'model.layers.{cfg.num_hidden_layers-1}.'
        extra={prefix+k.removeprefix(last):s for k,s in expected.items() if k.startswith(last)}
        extra.update({prefix+k:s for k,s in [('embed_tokens.weight',[cfg.vocab_size,cfg.hidden_size]),('shared_head.head.weight',[cfg.vocab_size,cfg.hidden_size]),('shared_head.norm.weight',[cfg.hidden_size]),('eh_proj.weight',[cfg.hidden_size,2*cfg.hidden_size]),('enorm.weight',[cfg.hidden_size]),('hnorm.weight',[cfg.hidden_size])]})
        expected.update(extra);excluded=set(extra)
    if set(entries)!=set(expected) or any(entries[k]['shape']!=s for k,s in expected.items()):reject('exact native text+declaredMTP inventory/shape required')
    routers=[k for k in entries if k.endswith(('.mlp.gate.weight','.mlp.gate.e_score_correction_bias'))];dtype=entries['model.embed_tokens.weight']['dtype']
    if dtype not in {'BF16','F32'} or any(v['dtype'] not in ({dtype,'F32'} if k in routers else {dtype}) for k,v in entries.items()):reject('homogeneous BF16/F32 weights and optional F32 routers')
    index=root/'model.safetensors.index.json'
    if index.exists():
        mapping=json.loads(index.read_text(encoding="utf-8")).get('weight_map')
        if not isinstance(mapping,dict) or set(mapping)!=set(entries) or any(Path(entries[k]['file']).name!=v for k,v in mapping.items()):reject('index/physical inventory mismatch')
    return {'architecture':'glm4_moe_lite','layers':cfg.num_hidden_layers,'config':cfg,'text_config':raw,'entries':entries,'required':sorted(entries),'active_required':sorted(active),'excluded_mtp':sorted(excluded),'expert_layout':'packed' if packed else 'individual','quantization_matrix_keys':[k for k in active if k.startswith('model.layers.') and k not in routers and len(entries[k]['shape'])in (2,3) and math.prod(entries[k]['shape'])>=128],'quantization_fp32_keys':[k for k in routers if k in active],'quantized_export':dict(CAPABILITY),'head_bytes':math.prod(entries['lm_head.weight']['shape'])*(2 if dtype=='BF16' else 4)}


def working_set(report,mode,rank,*,cuda):
    c=report['config'];active=report['active_required'];entries=report['entries'];items=[];slot=upload=0
    size=lambda k:estimated_bytes(report,mode,k)
    for i in range(report['layers']):
        keys=[k for k in active if k.startswith(f'model.layers.{i}.')];decoded=up=0
        for k in keys:
            v=entries[k];n=math.prod(v['shape']);quant=mode!='none' and k in report['quantization_matrix_keys']
            parts={'payload':((n+127)//128)*stride_for(mode),'scales':((n+127)//128)*4} if quant else {'value':size(k)}
            items.extend((nbytes,k,part) for part,nbytes in parts.items());up+=sum(parts.values())+511*len(parts)
            decoded+=n*(4 if (mode=='none' and v['dtype']=='F32') or (mode!='none' and k in report['quantization_fp32_keys']) else 2)+255
        slot=max(slot,decoded);upload=max(upload,up if mode!='none' else 0)
    capacities=[a['capacity'] for a in pinned_layout_items(items,512*1024**2)] if cuda else []
    emb=size('model.embed_tokens.weight');head=size('lm_head.weight')+size('model.norm.weight')
    lora=sum(rank*sum(entries[f'model.layers.{i}.{t}.weight']['shape'])*16 for i in range(c.num_hidden_layers) for t in TARGETS)
    native=slot;concat=0 if report['expert_layout']=='packed' else c.n_routed_experts*2*c.moe_intermediate_size*c.hidden_size*(4 if mode=='none' and entries['lm_head.weight']['dtype']=='F32' else 2)
    scratch=128*1024**2;maxsource=max(math.prod(v['shape'])*(4 if v['dtype']=='F32' else 2) for v in entries.values())
    host=(sum(capacities) if cuda else sum(size(k) for k in active if k.startswith('model.layers.')))+emb+max(maxsource+scratch,native+2*c.moe_intermediate_size*c.hidden_size*2+scratch)
    device=slot+upload+head+512*1024**2+concat+native+4*lora
    return {'host_before_payload':host,'device_before_payload':device,'pinned_capacities':capacities,'slot':slot,'upload':upload,'individual_concat':concat,'native_layer':native,'all8_lora_fields':lora,'source_max_tensor':maxsource}


class GlmLoader(WeightLoader):
    def read(self,key):
        if key in self.read_keys:raise ValueError('GLM repeated physical source read')
        if key not in self.report['excluded_mtp']:return super().read(key)
        # Exact inventory checked before any payload. Inactive MTP binds raw
        # source identity, never active storage/quantizer/pinned memory.
        with safe_open(self.report['entries'][key]['file'],framework='pt',device='cpu') as f:value=f.get_tensor(key).detach().contiguous()
        self.max_source_tensor_bytes=max(self.max_source_tensor_bytes,value.numel()*value.element_size());hash_tensor(self.original,key,value)
        if not finite(value):raise ValueError('nonfinite excluded MTP weight')
        self.read_keys.append(key);return None


class Glm4MoeLiteAdapter(DeepseekV3Adapter):
    def __init__(self,path,*,rank=2,alpha=4.,device='cpu',host_budget_bytes=1024**3,device_budget_bytes=2*1024**3,quantization='none',quantized_compute='bf16'):
        if type(rank)is not int or rank<1:raise ValueError('invalid LoRA rank before source loading')
        alpha=float(alpha)
        if not math.isfinite(alpha):raise ValueError('invalid LoRA alpha before source loading')
        self.device=torch.device(device)
        if self.device.type not in {'cpu','cuda'}:raise ValueError('GLM explicit CPU/CUDA execution only')
        report=inspect_glm4_moe_lite(path)
        if quantized_compute!='bf16' or quantization not in ('none',*FORMATS):raise ValueError('unsupported_quantization: explicit g128 storage/BF16 compute only')
        budget=working_set(report,quantization,rank,cuda=self.device.type=='cuda')
        if budget['host_before_payload']>host_budget_bytes or budget['device_before_payload']>device_budget_bytes:raise MemoryError('resource_budget: GLM complete pinned/source/scratch/slot/upload/head/concat/LoRA/native-oracle budget exceeds allowance before payload')
        before=source_snapshot(path,report['entries']);pin=[k for k in report['active_required'] if k.startswith('model.layers.')] if self.device.type=='cuda' else []
        loader=GlmLoader(path,report,quantization,pin_keys=pin,host_budget_bytes=host_budget_bytes,pinned_arena_limit=512*1024**2);state={};boundary={'model.norm.weight','lm_head.weight'}
        for k in report['required']:
            record=loader.read(k)
            if record is not None:state[k]=materialize(record,self.device) if k in boundary else record
        identity=loader.finish()
        if source_snapshot(path,report['entries'])!=before or scan_entries(path,'glm4_moe_lite')!=report['entries']:raise ValueError('GLM source mutated while loading')
        self.identity=hashlib.sha256((identity+RECIPE_SHA).encode()).hexdigest();self.source_identity=loader.source_identity;self.source_read_keys=list(loader.read_keys);self.pinned_owners=loader.pinned_owners
        self.config=report['config'];self.config.first_k_dense_replace=1;self.layout=report['expert_layout'];self.targets=list(TARGETS);self.excluded_mtp=list(report['excluded_mtp']);self.working_set=budget
        self.embedding=materialize(state.pop('model.embed_tokens.weight'));self.norm=state.pop('model.norm.weight');self.head=state.pop('lm_head.weight');self.blocks=[];c=self.config
        for i in range(c.num_hidden_layers):
            prefix=f'model.layers.{i}.';records={k.removeprefix(prefix):state.pop(k) for k in list(state) if k.startswith(prefix)}
            adapters=create_lora_for_block(i,'mla_moe',{k:v.shape for k,v in records.items()},rank,alpha,self.device,targets=TARGETS,architecture='glm4_moe_lite')
            parameters={'heads':c.num_attention_heads,'nope':c.qk_nope_head_dim,'rope':c.qk_rope_head_dim,'value':c.v_head_dim,'kv_rank':c.kv_lora_rank,'q_rank':c.q_lora_rank,'eps':c.rms_norm_eps,'attention_scale':(c.qk_nope_head_dim+c.qk_rope_head_dim)**-.5,'interleave':c.rope_interleave,'moe':c.mlp_layer_types[i]=='sparse','packed':self.layout=='packed','experts':c.n_routed_experts,'groups':c.n_group,'group_k':c.topk_group,'top_k':c.num_experts_per_tok,'normalize':c.norm_topk_prob,'router_scale':c.routed_scaling_factor}
            kwargs={'parameters':parameters,**dict(zip(['lora_qa','lora_qb','lora_kva','lora_kvb','lora_o'],adapters.values()))};tensors,specs=flatten_records(records)
            self.blocks.append(BlockPlan(i,'mla_moe',deepseek_block,kwargs,tensors,adapters,specs if quantization!='none' else None))
        assert not state
        self.storage_metadata=None if quantization=='none' else {'format':quantization,'compute':'bf16','storage_identity':loader.storage_identity,'source_identity':loader.source_identity,'total_stored_bytes':loader.stored_bytes,'max_source_tensor_bytes':loader.max_source_tensor_bytes,'full_bf16_decoder_retained':False,'quantized_matrices':loader.quantized_tensors,'decoder_stored_bytes':sum(t.numel()*t.element_size() for b in self.blocks for t in b.host_weights.values()),'cpu_embedding_bytes':self.embedding.numel()*self.embedding.element_size(),'device_boundary_bytes':self.head.numel()*self.head.element_size()+self.norm.numel()*self.norm.element_size(),'resident_decoder_core_bytes':0,'quantized_scope':'active text decoder matrices; F32 routers; source MTP raw hashed/discarded','excluded_mtp_keys':len(self.excluded_mtp),'recipe_sha256':RECIPE_SHA}

    def runtime_metadata(self):
        return {'architecture':'glm4_moe_lite','quantized_export':dict(CAPABILITY),'layer_count':len(self.blocks),'expert_layout':self.layout,'attention':'eager_MLA','expert_backend':'eager_per_expert_linear','rope':self.config.rope_parameters,'router':'frozen F32 sigmoid/noaux_tc with input VJP','lora_targets':self.targets,'objective':'causal_CE_only_explicit_excluded_MTP','excluded_mtp_key_count':len(self.excluded_mtp),'source_read_count':len(self.source_read_keys),'source_identity':self.source_identity,'recipe_sha256':RECIPE_SHA,'direct_pinned_capacity_bytes':sum(self.working_set['pinned_capacities']),'working_set':self.working_set,'native_lowbit_gemm':False,'kv_cache':'disabled','asynchronous_transfers':'unsupported','expert_scheduler':'all experts of one layer share one slot'}

    def close(self):
        super().close();self.pinned_owners.clear()
