"""Pure execution policies retain optimizer math and fail before mutation."""
import copy
from types import SimpleNamespace
import pytest
import torch
from ram_vram_finetune.config import TrainConfig
from ram_vram_finetune.memory import SlotSession
from ram_vram_finetune.finetune.lora import LoRAProjection
from ram_vram_finetune.finetune.optimizer import LoRAOptimizer

FIELDS=('A','grad_A','m_A','v_A','B','grad_B','m_B','v_B')

def projections():
    torch.manual_seed(837)
    result={}
    for name in ('first','last'):
        a=torch.randn(2,5);b=torch.randn(7,2)
        result[name]=LoRAProjection(name,2,4.,5,7,a,b,torch.randn_like(a),torch.randn_like(b),torch.zeros_like(a),torch.zeros_like(a),torch.zeros_like(b),torch.zeros_like(b))
    return result

def test_validation_execution_policy_preserves_exact_updates_and_state():
    scalar=projections();batched=copy.deepcopy(scalar)
    left=LoRAOptimizer(lr=.01,weight_decay=.1);right=LoRAOptimizer(lr=.01,weight_decay=.1,validation_mode='batched')
    for _ in range(3):
        left.step(scalar);right.step(batched)
        assert left.state_dict()==right.state_dict()
        for name in scalar:
            for field in FIELDS:assert torch.equal(getattr(scalar[name],field),getattr(batched[name],field))
    right.zero_grad(batched)
    assert all(torch.count_nonzero(p.grad_A)==0 and torch.count_nonzero(p.grad_B)==0 for p in batched.values())

@pytest.mark.parametrize('field',FIELDS)
@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_batched_rejects_nonfinite_any_field_without_partial_update(field,value):
    p=projections()
    with torch.no_grad():getattr(p['last'],field).flatten()[0]=value
    before=copy.deepcopy(p);opt=LoRAOptimizer(validation_mode='batched')
    with pytest.raises(FloatingPointError,match=field):opt.step(p)
    assert opt.step_count==0
    for name in p:
        for f in FIELDS:torch.testing.assert_close(getattr(p[name],f),getattr(before[name],f),rtol=0,atol=0,equal_nan=True)

@pytest.mark.parametrize('field',['v_A','v_B'])
def test_batched_rejects_negative_moments_without_partial_update(field):
    p=projections();getattr(p['last'],field).flatten()[0]=-1
    before=copy.deepcopy(p);opt=LoRAOptimizer(validation_mode='batched')
    with pytest.raises(ValueError,match='non-negative'):opt.step(p)
    assert opt.step_count==0
    for name in p:
        for f in FIELDS:assert torch.equal(getattr(p[name],f),getattr(before[name],f))

def test_batched_retains_shape_and_duplicate_checks():
    p=projections();p['last'].grad_A=torch.zeros(1)
    with pytest.raises(ValueError,match='shape'):LoRAOptimizer(validation_mode='batched').step(p)
    p=projections();p['alias']=p['first']
    with pytest.raises(ValueError,match='more than once'):LoRAOptimizer(validation_mode='batched').step(p)

def test_execution_policy_not_checkpoint_identity():
    base=TrainConfig('model','tokens','out',device='cuda:0')
    tuned=TrainConfig('model','tokens','out2',device='cuda:0',transfer_mode='batched',optimizer_validation='batched')
    assert base.semantics()==tuned.semantics()

@pytest.mark.parametrize('kw',[{'transfer_mode':'auto'}, {'optimizer_validation':'off'}, {'transfer_mode':'batched'}, {'device':'cuda:0','quantization':'q6','transfer_mode':'batched'}])
def test_unsupported_policy_is_explicit(kw):
    with pytest.raises(ValueError,match='unsupported_capability'):TrainConfig('model','data','out',**kw)

def test_cpu_slot_rejects_batched_without_allocating():
    with pytest.raises(ValueError,match='requires CUDA'):
        SlotSession([], 'cpu',host_budget_bytes=1,device_budget_bytes=1,transfer_mode='batched')
