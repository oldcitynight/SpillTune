"""CE chunking must retain native full-sequence terminal-normalization VJP."""
import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from ram_vram_finetune.training.engine import microbatch


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('chunk',[1,3,16])
@pytest.mark.parametrize('length',[5,8])
def test_full_sequence_norm_vjp_with_chunked_vocabulary_loss(dtype,chunk,length):
    torch.manual_seed(914);torch.set_num_threads(1)
    embedding=torch.randn(32,2048,dtype=dtype)*.1
    head=torch.randn(32,2048,dtype=dtype)*.01
    norm=Qwen3RMSNorm(2048).to(dtype=dtype).requires_grad_(False)
    with torch.no_grad():norm.weight.uniform_(.5,1.5)
    calls=[]
    class TerminalOnlyAdapter:
        device=torch.device('cpu');blocks=[]
        def positions_and_masks(self,valid):return (),{}
        def terminal_norm(self,x):calls.append(tuple(x.shape));return norm(x)
    adapter=TerminalOnlyAdapter();adapter.embedding=embedding;adapter.head=head
    ids=torch.tensor([[1,2,3,4,5,6,7,8]]);labels=torch.tensor([[-100,-100,-100,-100,-100,6,7,8]])
    ids=ids[:,:length];labels=labels[:,:length].clone()
    if length==5:labels[:,-1]=5
    denominator=int((labels[:,1:]!=-100).sum())
    x=F.embedding(ids,embedding).detach().requires_grad_(True)
    # Native full-sequence norm precedes selecting shifted-logit rows.
    logits=F.linear(norm(x)[:,:-1],head)
    loss=F.cross_entropy(logits.float().reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')/denominator
    expected,=torch.autograd.grad(loss,x)
    actual,gradient=microbatch(adapter,None,ids,labels,normalization=denominator,token_chunk=chunk)
    assert calls==[(1,length,2048)]
    assert actual==pytest.approx(float(loss.detach()),abs=3e-6,rel=3e-6)
    torch.testing.assert_close(gradient,expected,rtol=.02 if dtype==torch.bfloat16 else 2e-5,atol=2e-4 if dtype==torch.bfloat16 else 2e-6)
    relative=float((gradient-expected).float().norm())/float(expected.float().norm())
    assert relative<(.02 if dtype==torch.bfloat16 else 3e-5)
    if chunk==16:assert torch.equal(gradient,expected)
