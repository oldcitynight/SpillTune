"""Frozen canonical Qwen3.5 embedding/head; whole native HF training oracle.

RVFT_QWEN35_DEVICE is used only by the bounded, explicit tiny-GPU validator.
Default and restored test suites are CPU-only.
"""
import dataclasses
import json
import os
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from ram_vram_finetune.adapters.qwen35.adapter import Qwen35Adapter, inspect_model
from ram_vram_finetune.quantization.storage import WeightLoader, export_quantized
from ram_vram_finetune.training.engine import flatten, microbatch, run
from ram_vram_finetune.finetune.optimizer import LoRAOptimizer
from ram_vram_finetune.memory import SlotSession
from ram_vram_finetune.config import TrainConfig

DEVICE = os.environ.get('RVFT_QWEN35_DEVICE', 'cpu')


def make_source(path, dtype=torch.float32):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    torch.set_num_threads(1); torch.manual_seed(821); path.mkdir()
    cfg=Qwen3_5TextConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        linear_key_head_dim=4,linear_value_head_dim=4,linear_num_key_heads=2,
        linear_num_value_heads=2,linear_conv_kernel_dim=4,
        layer_types=['linear_attention']*3+['full_attention'],
        rope_parameters={'rope_type':'default','rope_theta':10000000.,
            'partial_rotary_factor':.5,'mrope_section':[1,1,0],'mrope_interleaved':True},
        attention_dropout=0.,use_cache=False,tie_word_embeddings=True)
    cfg._attn_implementation='eager'
    native=Qwen3_5TextModel(cfg).eval()
    # Nonzero offset norms exercise Qwen3.5's 1+weight semantics.
    source={'model.language_model.'+k:v.detach().clone().to(dtype) for k,v in native.state_dict().items()}
    for key,value in source.items():
        if key.endswith(('q_norm.weight','k_norm.weight','input_layernorm.weight','post_attention_layernorm.weight','norm.weight')):
            value.uniform_(-.15,.15)
    save_file(source,str(path/'model.safetensors'))
    (path/'config.json').write_text(json.dumps({'model_type':'qwen3_5','tie_word_embeddings':True,'text_config':cfg.to_dict()}))
    rows=[{'input_ids':[1,2,3,4,5],'labels':[-100,-100,3,4,5]},
          {'input_ids':[3,6,9],'labels':[-100,-100,9]}]
    (path/'tokens.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def setup(path,dtype=torch.float32):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    make_source(path,dtype)
    adapter=Qwen35Adapter(path,device=DEVICE)
    raw=json.loads((path/'config.json').read_text())
    cfg=Qwen3_5TextConfig(**raw['text_config']);cfg._attn_implementation='eager'
    with torch.device('meta'):native=Qwen3_5TextModel(cfg)
    state={k.removeprefix('model.language_model.'):v for k,v in load_file(str(path/'model.safetensors')).items()}
    native.load_state_dict(state,strict=True,assign=True)
    # Do not cast the model: native RoPE buffers must stay FP32.
    native.rotary_emb=__import__('transformers.models.qwen3_5.modeling_qwen3_5',fromlist=['Qwen3_5TextRotaryEmbedding']).Qwen3_5TextRotaryEmbedding(cfg)
    native.to(DEVICE).eval().requires_grad_(False)
    assert native.rotary_emb.inv_freq.dtype==torch.float32
    refs={}
    class Projection(torch.nn.Module):
        def __init__(self,base,p):
            super().__init__();self.base=base;self.A=torch.nn.Parameter(p.A.detach().clone());self.B=torch.nn.Parameter(p.B.detach().clone());self.scale=p.scale
        def forward(self,x):
            base=self.base(x)
            return (base.float()+F.linear(F.linear(x.float(),self.A),self.B)*self.scale).to(base.dtype)
    for b in adapter.blocks:
        for target,p in b.adapters.items():
            parent,leaf=target.rsplit('.',1);owner=native.layers[b.index].get_submodule(parent)
            ref=Projection(getattr(owner,leaf),p);setattr(owner,leaf,ref);refs[f'{b.index}:{target}']=ref
    slot=SlotSession(adapter.blocks,DEVICE,host_budget_bytes=1024**3,device_budget_bytes=1024**3)
    return adapter,native,refs,slot


def native_loss(native,ids,labels):
    x=native.embed_tokens(ids.to(DEVICE)).detach().requires_grad_(True)
    hidden=native(inputs_embeds=x,use_cache=False).last_hidden_state
    # One physical frozen Parameter supplies both embedding and output projection.
    logits=F.linear(hidden,native.embed_tokens.weight)
    loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),labels[:,1:].to(DEVICE).reshape(-1),ignore_index=-100,reduction='sum')
    return loss,x


def compare(a,b,dtype):
    torch.testing.assert_close(a,b,rtol=.06 if dtype==torch.bfloat16 else 3e-5,atol=.003 if dtype==torch.bfloat16 else 3e-6)
    relative=float((a-b).detach().float().norm())/max(float(b.detach().float().norm()),1e-12)
    assert relative<(.08 if dtype==torch.bfloat16 else 3e-4)
    return float((a-b).detach().abs().max()),relative


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('nonzero_b',[False,True])
def test_whole_hf_loss_input_and_all_gradients(tmp_path,dtype,nonzero_b,record_property):
    adapter,native,refs,slot=setup(tmp_path/'model',dtype)
    if nonzero_b:
        with torch.no_grad():
            for key,p in flatten(adapter).items():p.B.normal_(0,.03);refs[key].B.copy_(p.B)
    ids=torch.tensor([[1,3,5,7,9]]);labels=torch.tensor([[-100,-100,5,7,9]])
    loss,x=native_loss(native,ids,labels);(loss/3).backward()
    with slot.run():actual,gradient=microbatch(adapter,slot,ids,labels,normalization=3,token_chunk=2)
    assert actual==pytest.approx(float(loss.detach()/3),abs=3e-6,rel=3e-6)
    errors=[compare(gradient,x.grad,dtype)]
    for key,p in flatten(adapter).items():
        errors.extend([compare(p.grad_A,refs[key].A.grad,dtype),compare(p.grad_B,refs[key].B.grad,dtype)])
    assert torch.count_nonzero(gradient[:,:2])>0
    assert adapter.head.requires_grad is False and native.embed_tokens.weight.grad is None
    assert torch.equal(adapter.embedding,adapter.head.cpu())
    if DEVICE=='cpu':assert adapter.embedding.data_ptr()==adapter.head.data_ptr()
    else:assert adapter.embedding.device.type=='cpu' and adapter.head.device.type=='cuda'
    record_property('max_gradient_abs_error',max(x[0] for x in errors));record_property('max_gradient_relative_l2',max(x[1] for x in errors));record_property('loss_abs_error',abs(actual-float(loss.detach()/3)))
    assert slot.reloads==8
    slot.close();adapter.close()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_two_updates_unequal_accumulation_and_moments(tmp_path,dtype,record_property):
    adapter,native,refs,slot=setup(tmp_path/'model',dtype)
    with torch.no_grad():
        for key,p in flatten(adapter).items():p.B.normal_(0,.02);refs[key].B.copy_(p.B)
    optimizer=LoRAOptimizer(lr=.001,weight_decay=.1)
    oracle=torch.optim.AdamW([t for p in refs.values() for t in (p.A,p.B)],lr=.001,weight_decay=.1,foreach=False)
    rows=[(torch.tensor([[1,2,3,4,5]]),torch.tensor([[-100,-100,3,4,5]])),(torch.tensor([[3,6,9]]),torch.tensor([[-100,-100,9]]))]
    frozen={f'{b.index}:{k}':v.clone() for b in adapter.blocks for k,v in b.host_weights.items()}
    embedding=adapter.embedding.clone();head=adapter.head.clone();norm=adapter.norm.clone()
    errors=[];update_errors=[]
    for _ in range(2):
        oracle.zero_grad(set_to_none=True);optimizer.zero_grad(flatten(adapter))
        loss=sum(native_loss(native,*row)[0] for row in rows)/4;loss.backward()
        with slot.run():actual=sum(microbatch(adapter,slot,*row,normalization=4,token_chunk=2,accumulate=i>0)[0] for i,row in enumerate(rows))
        assert actual==pytest.approx(float(loss.detach()),abs=3e-6,rel=3e-6)
        for key,p in flatten(adapter).items():
            errors.extend([compare(p.grad_A,refs[key].A.grad,dtype),compare(p.grad_B,refs[key].B.grad,dtype)])
            # BF16: compare independently obtained gradients before same-gradient
            # AdamW protocol. FP32 uses the independent full-HF trajectory.
            if dtype==torch.bfloat16:refs[key].A.grad.copy_(p.grad_A);refs[key].B.grad.copy_(p.grad_B)
        optimizer.step(flatten(adapter));oracle.step()
        for key,p in flatten(adapter).items():
            for field in ('A','B'):
                update_errors.append(compare(getattr(p,field),getattr(refs[key],field),torch.float32)[0])
                for prefix,name in [('m_','exp_avg'),('v_','exp_avg_sq')]:update_errors.append(compare(getattr(p,prefix+field),oracle.state[getattr(refs[key],field)][name],torch.float32)[0])
    assert all(torch.equal(v,frozen[f'{b.index}:{k}']) for b in adapter.blocks for k,v in b.host_weights.items())
    assert torch.equal(embedding,adapter.embedding) and torch.equal(head,adapter.head) and torch.equal(norm,adapter.norm)
    record_property('max_gradient_abs_error',max(x[0] for x in errors));record_property('max_gradient_relative_l2',max(x[1] for x in errors));record_property('max_optimizer_error',max(update_errors))
    slot.close();adapter.close()


def test_alias_read_once_budget_and_excluded_roles(tmp_path):
    path=make_source(tmp_path/'model');report=inspect_model(path)
    assert report['source_aliases']=={'lm_head.weight':'model.language_model.embed_tokens.weight'}
    reads=[];original=WeightLoader.read
    def observed(self,key):reads.append(key);return original(self,key)
    with patch.object(WeightLoader,'read',observed):adapter=Qwen35Adapter(path)
    assert reads==report['required'] and reads.count('model.language_model.embed_tokens.weight')==1 and 'lm_head.weight' not in reads
    assert adapter.embedding.data_ptr()==adapter.head.data_ptr();adapter.close()
    for kw in [{'host_budget_bytes':1},{'device_budget_bytes':report['head_bytes']-1}]:
        with patch.object(WeightLoader,'read',side_effect=AssertionError('payload read')):
            with pytest.raises(MemoryError,match='resource_budget'):Qwen35Adapter(path,**kw)
    state=load_file(str(path/'model.safetensors'));state['mtp.fc.weight']=torch.ones(2,2);state['model.visual.example.weight']=torch.ones(2,2);save_file(state,str(path/'model.safetensors'))
    report=inspect_model(path);assert {k:len(v) for k,v in report['excluded_nontext_tensors'].items()}=={'vision':1,'mtp':1}
    with patch.object(WeightLoader,'read',observed):adapter=Qwen35Adapter(path)
    assert not any(k.startswith(('mtp.','model.visual.')) for k in reads)
    assert adapter.runtime_metadata()['excluded_nontext_tensor_counts']=={'vision':1,'mtp':1};adapter.close()


@pytest.mark.parametrize('change',[{'tie_word_embeddings':1},{'tie_word_embeddings':'true'},{'tie_word_embeddings':False}])
def test_malformed_or_conflicting_tie_rejected_before_payload(tmp_path,change):
    path=make_source(tmp_path/'model');raw=json.loads((path/'config.json').read_text());raw.update(change);(path/'config.json').write_text(json.dumps(raw))
    with patch.object(WeightLoader,'read',side_effect=AssertionError('payload read')):
        with pytest.raises(ValueError,match='unsupported_capability'):Qwen35Adapter(path)


def test_duplicate_or_unknown_tensor_rejected_before_payload(tmp_path):
    path=make_source(tmp_path/'model');state=load_file(str(path/'model.safetensors'))
    for key in ['lm_head.weight','unrecognized.weight','model.language_model.other.weight']:
        altered={**state,key:state['model.language_model.embed_tokens.weight'].clone()};save_file(altered,str(path/'model.safetensors'))
        with patch.object(WeightLoader,'read',side_effect=AssertionError('payload read')):
            with pytest.raises(ValueError,match='unsupported_capability'):Qwen35Adapter(path)


@pytest.mark.parametrize('mode',['fp8-e4m3fn-g128','fp8-e5m2-g128','int8-sym-g128','int6-sym-g128'])
def test_compressed_tied_source_explicitly_rejected_before_payload(tmp_path,mode):
    path=make_source(tmp_path/'model')
    with patch.object(WeightLoader,'read',side_effect=AssertionError('payload read')):
        with pytest.raises(ValueError,match='unsupported_quantization'):Qwen35Adapter(path,quantization=mode)
        with pytest.raises(ValueError,match='unsupported_quantization'):export_quantized(path,tmp_path/'archive','qwen35',mode)
    assert not (tmp_path/'archive').exists()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_checkpoint_state_rng_metadata_exports_and_resume(tmp_path,dtype):
    path=make_source(tmp_path/'model',dtype)
    config=TrainConfig(str(path),str(path/'tokens.jsonl'),str(tmp_path/'full'),steps=2,accumulation=2,device=DEVICE)
    full=run(config)
    cancelled=run(dataclasses.replace(config,output=str(tmp_path/'cancel')),cancel_after_step=1)
    resumed=run(dataclasses.replace(config,output=str(tmp_path/'resumed')),resume=tmp_path/'cancel/checkpoint-1')
    a=load_file(str(tmp_path/'full/checkpoint-2/state.safetensors'));b=load_file(str(tmp_path/'resumed/checkpoint-2/state.safetensors'))
    assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    assert json.loads((tmp_path/'full/checkpoint-2/metadata.json').read_text())==json.loads((tmp_path/'resumed/checkpoint-2/metadata.json').read_text())
    assert full['status']==resumed['status']=='completed' and cancelled['status']=='cancelled'
    for folder,file in [('adapter','state.safetensors'),('peft-adapter','adapter_model.safetensors')]:
        a=load_file(str(tmp_path/'full'/folder/file));b=load_file(str(tmp_path/'resumed'/folder/file))
        assert len(a)==12 and a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)


def test_untied_omitted_default_preserves_historical_identity(tmp_path):
    from ram_vram_finetune.fixture import make_tiny
    from ram_vram_finetune.quantization.storage import hash_tensor
    import hashlib
    path=make_tiny(tmp_path/'model');raw=json.loads((path/'config.json').read_text());raw.pop('tie_word_embeddings',None)
    (path/'config.json').write_text(json.dumps(raw));report=inspect_model(path)
    assert report['text_config']==raw and 'tie_word_embeddings' not in report['text_config']
    expected=hashlib.sha256();expected.update(json.dumps(raw,sort_keys=True).encode())
    state=load_file(str(path/'model.safetensors'))
    for key in report['required']:hash_tensor(expected,key,state[key])
    adapter=Qwen35Adapter(path);assert adapter.identity==expected.hexdigest();adapter.close()


def test_top_level_only_tied_declaration_is_resolved(tmp_path):
    path=make_source(tmp_path/'model');raw=json.loads((path/'config.json').read_text());raw['text_config'].pop('tie_word_embeddings')
    (path/'config.json').write_text(json.dumps(raw));adapter=Qwen35Adapter(path)
    assert adapter.config.tie_word_embeddings is True and adapter.embedding.data_ptr()==adapter.head.data_ptr();adapter.close()
