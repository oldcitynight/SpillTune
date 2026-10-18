"""Native Qwen3 and frozen tied-boundary contracts; no downloaded weights."""
import json
from unittest.mock import patch
import pytest
import torch
from safetensors.torch import load_file, save_file
from ram_vram_finetune.adapters.dense import inspect_dense,Qwen3Adapter
from ram_vram_finetune.fixture import make_tiny
from ram_vram_finetune.quantization.storage import WeightLoader,export_quantized
from test_dense_adapters import setup,close,native_loss
from ram_vram_finetune.training.engine import flatten,microbatch
from ram_vram_finetune.finetune.optimizer import LoRAOptimizer


def test_frozen_alias_read_once_and_cpu_storage_identity(tmp_path):
    path=make_tiny(tmp_path/'model',adapter='qwen3',layers=2)
    report=inspect_dense(path,'qwen3');assert report['source_aliases']=={'lm_head.weight':'model.embed_tokens.weight'}
    assert 'lm_head.weight' not in report['entries'] and report['head_bytes']==32*16*4
    original=WeightLoader.read;seen=[]
    def observe(loader,key):seen.append(key);return original(loader,key)
    with patch.object(WeightLoader,'read',observe):adapter=Qwen3Adapter(path)
    assert seen.count('model.embed_tokens.weight')==1 and 'lm_head.weight' not in seen
    assert adapter.embedding.data_ptr()==adapter.head.data_ptr()
    assert not adapter.embedding.requires_grad and not adapter.head.requires_grad
    assert adapter.runtime_metadata()['shared_weights']=='frozen_embedding_head_source_alias'
    before=adapter.identity
    # Source alias is bound through config and physical tensors, with no synthetic head tensor.
    assert len(before)==64 and len(adapter.blocks)==2
    adapter.close()


@pytest.mark.parametrize('change',[{'attention_bias':True},{'sliding_window':2,'use_sliding_window':True},
    {'tie_word_embeddings':1},{'rope_parameters':{'rope_type':'linear','rope_theta':10000.,'factor':2.}}])
def test_qwen3_reject_before_payload(tmp_path,change):
    path=make_tiny(tmp_path/'model',adapter='qwen3',layers=2)
    raw=json.loads((path/'config.json').read_text());raw.update(change);(path/'config.json').write_text(json.dumps(raw))
    with patch.object(WeightLoader,'read',side_effect=AssertionError('payload must not be read')):
        with pytest.raises(ValueError,match='unsupported_capability'):Qwen3Adapter(path)


@pytest.mark.parametrize('mutation',['duplicate_head','missing_q_norm','wrong_k_norm_shape'])
def test_qwen3_strict_tied_inventory(tmp_path,mutation):
    path=make_tiny(tmp_path/'model',adapter='qwen3',layers=2);state=load_file(str(path/'model.safetensors'))
    if mutation=='duplicate_head':state['lm_head.weight']=state['model.embed_tokens.weight'].clone()
    elif mutation=='missing_q_norm':state.pop('model.layers.0.self_attn.q_norm.weight')
    else:state['model.layers.0.self_attn.k_norm.weight']=torch.ones(7)
    save_file(state,str(path/'model.safetensors'))
    with patch.object(WeightLoader,'read',side_effect=AssertionError('payload must not be read')):
        with pytest.raises(ValueError,match='inventory|shape mismatch'):Qwen3Adapter(path)


def test_tied_head_budget_and_unvalidated_quant_reject_early(tmp_path):
    path=make_tiny(tmp_path/'model',adapter='qwen3',layers=2)
    with patch.object(WeightLoader,'read',side_effect=AssertionError('payload must not be read')):
        with pytest.raises(MemoryError,match='budget'):Qwen3Adapter(path,device_budget_bytes=100)
        with pytest.raises(ValueError,match='unsupported_quantization'):Qwen3Adapter(path,quantization='int6-sym-g128')
        with pytest.raises(ValueError,match='unsupported_quantization'):export_quantized(path,tmp_path/'archive','qwen3','int6-sym-g128')
    assert not (tmp_path/'archive').exists()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_qwen3_nonzero_b_two_steps_all_gradients_and_moments(tmp_path,dtype):
    _,adapter,native,refs,slot=setup(tmp_path,'qwen3',dtype=dtype);projections=flatten(adapter)
    with torch.no_grad():
        for k,p in projections.items():p.B.normal_(0,.03);refs[k].B.copy_(p.B)
    optimizer=LoRAOptimizer(lr=.001,weight_decay=.1)
    reference_optimizer=torch.optim.AdamW([v for ref in refs.values() for v in (ref.A,ref.B)],lr=.001,weight_decay=.1,foreach=False)
    rows=[(torch.tensor([[1,2,3,4,5]]),torch.tensor([[-100,-100,3,4,5]])),(torch.tensor([[3,6,9]]),torch.tensor([[-100,-100,9]]))]
    frozen=adapter.embedding.clone()
    for step in range(2):
        reference_optimizer.zero_grad(set_to_none=True);optimizer.zero_grad(projections)
        loss=sum(native_loss(native,*row)[0] for row in rows)/4;loss.backward()
        with slot.run():actual=sum(microbatch(adapter,slot,*row,normalization=4,token_chunk=2,accumulate=i>0)[0] for i,row in enumerate(rows))
        assert actual==pytest.approx(float(loss.detach()),abs=3e-6,rel=3e-6)
        for key,p in projections.items():
            close(p.grad_A,refs[key].A.grad,dtype);close(p.grad_B,refs[key].B.grad,dtype)
            for field in ('A','B'):
                left=getattr(p,'grad_'+field);right=getattr(refs[key],field).grad
                relative=float((left-right).float().norm())/max(float(right.float().norm()),1e-12)
                assert relative < (.08 if dtype==torch.bfloat16 else 3e-4), (step,key,field,relative)
            if dtype==torch.bfloat16:
                # Compare native gradients first, then exact same-gradient optimizer protocol.
                refs[key].A.grad.copy_(p.grad_A);refs[key].B.grad.copy_(p.grad_B)
        optimizer.step(projections);reference_optimizer.step()
        for key,p in projections.items():
            for name in ('A','B'):
                if dtype==torch.bfloat16:assert torch.equal(getattr(p,name),getattr(refs[key],name))
                else:close(getattr(p,name),getattr(refs[key],name))
                for prefix,field in [('m_','exp_avg'),('v_','exp_avg_sq')]:close(getattr(p,prefix+name),reference_optimizer.state[getattr(refs[key],name)][field])
    assert torch.equal(adapter.embedding,frozen) and torch.equal(adapter.head,frozen)
    slot.close();adapter.close()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_untied_qwen3_independent_full_hf(tmp_path,dtype):
    _,adapter,native,refs,slot=setup(tmp_path,'qwen3',dtype=dtype,config_patch={'tie_word_embeddings':False})
    assert not adapter.source_aliases and adapter.embedding.data_ptr()!=adapter.head.data_ptr()
    with torch.no_grad():
        for key,p in flatten(adapter).items():p.B.normal_(0,.02);refs[key].B.copy_(p.B)
    ids=torch.tensor([[1,3,5,7,9]]);labels=torch.tensor([[-100,-100,5,7,9]])
    loss,x=native_loss(native,ids,labels);(loss/3).backward()
    with slot.run():actual,grad=microbatch(adapter,slot,ids,labels,normalization=3,token_chunk=2)
    assert actual==pytest.approx(float(loss.detach()/3),abs=3e-6,rel=3e-6);close(grad,x.grad,dtype)
    for key,p in flatten(adapter).items():close(p.grad_A,refs[key].A.grad,dtype);close(p.grad_B,refs[key].B.grad,dtype)
    slot.close();adapter.close()
