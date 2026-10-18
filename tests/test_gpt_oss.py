"""Native decoded HF oracle, gradients, routing negatives and source refusals."""
import dataclasses
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file,load_file
from ram_vram_finetune.adapters.gpt_oss.adapter import GptOssAdapter,NativeLoader,inspect_gpt_oss,hf_classes
from ram_vram_finetune.adapters.gpt_oss.fixture import state
from ram_vram_finetune.adapters.gpt_oss.mxfp4 import decode_expert,validate_pair,VALUES
from ram_vram_finetune.adapters.gpt_oss.ops import feed_forward,router
from ram_vram_finetune.memory import SlotSession
from ram_vram_finetune.training.engine import flatten,microbatch,run
from ram_vram_finetune.config import TrainConfig
from ram_vram_finetune.finetune.optimizer import LoRAOptimizer
from ram_vram_finetune.quantization.storage import export_quantized


def official(blocks,scales):
    from transformers.integrations.mxfp4 import _convert_moe_packed_tensors
    return _convert_moe_packed_tensors(blocks,scales)


def setup(path,dtype=torch.float32,top_k=2):
    torch.set_num_threads(1);torch.manual_seed(731);path.mkdir();cfg,values=state(2,dtype=dtype,top_k=top_k)
    save_file(values,str(path/'model.safetensors'));(path/'config.json').write_text(json.dumps(cfg.to_dict()))
    (path/'tokens.jsonl').write_text(json.dumps({'input_ids':[1,2,3,4,5],'labels':[-100,-100,3,4,5]})+'\n'+json.dumps({'input_ids':[3,6,9],'labels':[-100,6,9]})+'\n')
    adapter=GptOssAdapter(path);_,_,Model=hf_classes();native_values={k:v.clone() for k,v in values.items() if not k.endswith(('_blocks','_scales'))}
    for key in values:
        if key.endswith('_blocks'):native_values[key.removesuffix('_blocks')]=official(values[key],values[key.removesuffix('_blocks')+'_scales']).to(dtype)
    native=Model(cfg).eval().requires_grad_(False);native.load_state_dict(native_values,strict=True,assign=True)
    refs={}
    class Projection(torch.nn.Module):
        def __init__(self,base,p):
            super().__init__();self.base=base;self.A=torch.nn.Parameter(p.A.detach().clone());self.B=torch.nn.Parameter(p.B.detach().clone());self.scale=p.scale
        def forward(self,x):
            y=self.base(x);return (y.float()+F.linear(F.linear(x.float(),self.A),self.B)*self.scale).to(y.dtype)
    for b in adapter.blocks:
        for target,p in b.adapters.items():
            parent,leaf=target.rsplit('.',1);owner=native.model.layers[b.index].get_submodule(parent);ref=Projection(getattr(owner,leaf),p);setattr(owner,leaf,ref);refs[f'{b.index}:{target}']=ref
    slot=SlotSession(adapter.blocks,'cpu',host_budget_bytes=1024**3,device_budget_bytes=1024**3)
    return adapter,native,refs,slot


def loss(native,ids,labels):
    x=native.model.embed_tokens(ids).detach().requires_grad_(True);out=native(inputs_embeds=x,use_cache=False,output_router_logits=False)
    assert out.aux_loss is None
    return F.cross_entropy(out.logits[:,:-1].float().reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum'),x


def compare(a,b,dtype=torch.float32):
    torch.testing.assert_close(a,b,atol=.004 if dtype==torch.bfloat16 else 4e-6,rtol=.07 if dtype==torch.bfloat16 else 4e-5)
    assert float((a-b).detach().float().norm())/max(float(b.detach().float().norm()),1e-12)<(.08 if dtype==torch.bfloat16 else .001)


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
@pytest.mark.parametrize('chunk',[1,7,65536])
def test_decode_all_nibbles_signed_zero_layout_and_chunk(dtype,chunk):
    blocks=torch.arange(256,dtype=torch.uint8).reshape(2,4,2,16);scales=torch.arange(16,dtype=torch.uint8).reshape(2,4,2)+118
    expected=official(blocks,scales).to(dtype)
    actual=torch.stack([decode_expert(b,s,dtype=dtype,chunk_groups=chunk) for b,s in zip(blocks,scales)])
    assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))


def test_finite_validator_exhaustive_256_scales_16_codes():
    # Cross product includes every exponent/code; exponent255 is E8M0 NaN,
    # rejected independently of torch.ldexp's code-zero implementation quirk.
    codes=torch.arange(16,dtype=torch.uint8);packed=(codes|(codes<<4))[None,:,None,None].expand(256,16,1,16).contiguous()
    scales=torch.arange(256,dtype=torch.uint8)[:,None,None].expand(256,16,1).contiguous()
    decoded=official(packed,scales)
    finite=torch.isfinite(decoded).all(dim=1)
    for s in range(256):
        for c in range(16):
            should=(s!=255 and bool(finite[s,c]))
            try:validate_pair(packed[s,c:c+1],scales[s,c:c+1],chunk_groups=1);accepted=True
            except ValueError:accepted=False
            assert accepted==should,(s,c)


@pytest.mark.parametrize('scale,code',[(255,0),(254,4),(254,12),(253,6),(253,15)])
def test_tail_corruption_not_sampled(scale,code):
    b=torch.zeros(1,20,1,16,dtype=torch.uint8);s=torch.full((1,20,1),120,dtype=torch.uint8);b[0,-1,0,-1]=code<<4;s[0,-1,0]=scale
    with pytest.raises(ValueError):validate_pair(b,s,chunk_groups=7)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('nonzero',[False,True])
@pytest.mark.parametrize('top_k',[1,2,4])
def test_native_whole_loss_prompt_and_all_lora_vjp(tmp_path,dtype,nonzero,top_k):
    a,n,refs,slot=setup(tmp_path/'source',dtype,top_k)
    if nonzero:
        with torch.no_grad():
            for key,p in flatten(a).items():p.B.normal_(0,.03);refs[key].B.copy_(p.B)
    ids=torch.tensor([[1,2,3,4,5],[2,6,9,0,0]]);labels=torch.tensor([[-100,-100,3,4,5],[-100,6,9,-100,-100]])
    # Separate unequal rows keep native mask independent and avoid a padded CE branch.
    expected=0.;expected_grads=[]
    for i,length in enumerate((5,3)):
        ell,x=loss(n,ids[i:i+1,:length],labels[i:i+1,:length]);(ell/5).backward();expected+=float(ell.detach()/5);expected_grads.append(x.grad)
    with slot.run():
        for i,length in enumerate((5,3)):
            actual,g=microbatch(a,slot,ids[i:i+1,:length],labels[i:i+1,:length],normalization=5,token_chunk=2,accumulate=i>0)
            compare(g,expected_grads[i],dtype)
            if i==0:assert g[:,:2].abs().sum()>0
            if i==0:total=actual
            else:total+=actual
    assert total==pytest.approx(expected,abs=3e-6,rel=3e-6)
    for key,p in flatten(a).items():compare(p.grad_A,refs[key].A.grad,dtype);compare(p.grad_B,refs[key].B.grad,dtype)
    assert len(a.source_read_keys)==len(set(a.source_read_keys))==41
    slot.close();a.close()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_two_updates_all_moments_and_frozen_source(tmp_path,dtype):
    a,n,refs,slot=setup(tmp_path/'source',dtype,4);opt=LoRAOptimizer(lr=.001,weight_decay=.1)
    with torch.no_grad():
        for key,p in flatten(a).items():p.B.normal_(0,.02);refs[key].B.copy_(p.B)
    oracle=torch.optim.AdamW([t for p in refs.values() for t in (p.A,p.B)],lr=.001,weight_decay=.1,foreach=False)
    rows=[(torch.tensor([[1,2,3,4,5]]),torch.tensor([[-100,-100,3,4,5]])),(torch.tensor([[3,6,9]]),torch.tensor([[-100,6,9]]))]
    frozen={f'{b.index}:{k}':v.clone() for b in a.blocks for k,v in b.host_weights.items()}
    for step in range(2):
        oracle.zero_grad();opt.zero_grad(flatten(a));expected=sum(loss(n,*row)[0] for row in rows)/5;expected.backward()
        with slot.run():actual=sum(microbatch(a,slot,*row,normalization=5,token_chunk=2,accumulate=i>0)[0] for i,row in enumerate(rows))
        assert actual==pytest.approx(float(expected.detach()),abs=3e-6,rel=3e-6)
        for key,p in flatten(a).items():
            compare(p.grad_A,refs[key].A.grad,dtype);compare(p.grad_B,refs[key].B.grad,dtype)
            # BF16 roundoff paths compared above; Adam oracle uses the SAME
            # actual gradients, not an independent native BF16 trajectory claim.
            if dtype==torch.bfloat16:refs[key].A.grad.copy_(p.grad_A);refs[key].B.grad.copy_(p.grad_B)
        opt.step(flatten(a));oracle.step()
        for key,p in flatten(a).items():
            for field in ('A','B'):
                compare(getattr(p,field),getattr(refs[key],field))
                for prefix,name in [('m_','exp_avg'),('v_','exp_avg_sq')]:compare(getattr(p,prefix+field),oracle.state[getattr(refs[key],field)][name])
    assert all(torch.equal(v,frozen[f'{b.index}:{k}']) for b in a.blocks for k,v in b.host_weights.items())
    slot.close();a.close()


def test_router_all_hit_bias_input_gradient_and_detach_negative(tmp_path):
    a,n,refs,slot=setup(tmp_path/'source',top_k=4);x=torch.randn(2,5,32,requires_grad=True);w=a.blocks[0].host_weights
    actual=feed_forward(x,w,experts=4,top_k=4);expected,_=n.model.layers[0].mlp(x);compare(actual,expected)
    ga,=torch.autograd.grad(actual.square().sum(),x,retain_graph=True);ge,=torch.autograd.grad(expected.square().sum(),x);compare(ga,ge)
    scores,indices=router(x.reshape(-1,32),w,4);native_scores,native_indices=n.model.layers[0].mlp.router(x.reshape(-1,32))[1:]
    assert set(indices.flatten().tolist())==set(range(4));assert torch.equal(indices,native_indices);compare(scores,native_scores)
    detached=n.model.layers[0].mlp.experts(x.reshape(-1,32),indices,scores.detach());gd,=torch.autograd.grad(detached.square().sum(),x)
    assert float((ge-gd).norm())/float(ge.norm())>1e-3
    assert w['self_attn.sinks'].abs().sum()>0 and w['mlp.router.bias'].abs().sum()>0 and w['mlp.experts.down_proj_bias'].abs().sum()>0
    slot.close();a.close()


def test_mask_right_padding_sliding_and_native_yarn_long_positions(tmp_path):
    a,n,refs,slot=setup(tmp_path/'source');valid=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool);rope,m=a.positions_and_masks(valid)
    assert m['sliding_attention'][0,0,4,2]<-1e10 and m['sliding_attention'][0,0,4,3]==0
    assert m['full_attention'][0,0,4,0]==0 and m['full_attention'][1,0,4,3]<-1e10
    for x,y in zip(rope,n.model.rotary_emb(torch.zeros(2,5,32),torch.arange(5).expand(2,-1))):assert torch.equal(x,y)
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
    Config,_,_=hf_classes();cfg=Config(head_dim=64,rope_parameters={'rope_type':'yarn','rope_theta':150000.,'factor':32.,'original_max_position_embeddings':4096,'beta_fast':32.,'beta_slow':1.,'truncate':False})
    r=GptOssRotaryEmbedding(cfg);positions=torch.tensor([[0,4095,4096,100000]]);cos,sin=r(torch.zeros((),dtype=torch.bfloat16),positions)
    # Independent YaRN frequency interpolation, no production helper.
    import math
    dim=64;theta=150000.;factor=32.;pos_freq=theta**(torch.arange(0,dim,2).float()/dim)
    lo=dim*math.log(4096/(32*2*math.pi))/(2*math.log(theta));hi=dim*math.log(4096/(1*2*math.pi))/(2*math.log(theta))
    ramp=((torch.arange(dim//2).float()-max(lo,0))/(min(hi,dim-1)-max(lo,0))).clamp(0,1)
    inv=(1/pos_freq)*(1-ramp)+(1/(factor*pos_freq))*ramp;phase=positions.float()[:,:,None]*inv;scale=.1*math.log(factor)+1
    assert torch.equal(cos,(phase.cos()*scale).bfloat16()) and torch.equal(sin,(phase.sin()*scale).bfloat16())
    slot.close();a.close()


@pytest.mark.parametrize('change',[{'output_router_logits':True},{'use_cache':True},{'swiglu_limit':8},{'tie_word_embeddings':True},{'attention_bias':False},{'attention_dropout':.1},{'experts_per_token':1},{'num_experts_per_tok':5},{'hidden_size':31},{'rope_scaling':{'rope_type':'dynamic','factor':2.}},{'quantization_config':{'quant_method':'int4'}}])
def test_metadata_refusal_before_payload(tmp_path,change):
    path=tmp_path/'source';a,n,r,s=setup(path);s.close();a.close();cfg=json.loads((path/'config.json').read_text());cfg.update(change);(path/'config.json').write_text(json.dumps(cfg))
    with patch.object(NativeLoader,'read',side_effect=AssertionError('payload forbidden')):
        with pytest.raises((ValueError,TypeError)):GptOssAdapter(path)


@pytest.mark.parametrize('mutation',['missing','wrong_shape','wrong_dtype','extra'])
def test_inventory_and_budget_before_payload(tmp_path,mutation):
    path=tmp_path/'source';a,n,r,s=setup(path);s.close();a.close();v=load_file(str(path/'model.safetensors'));k='model.layers.0.mlp.experts.down_proj_blocks'
    if mutation=='missing':v.pop(k)
    elif mutation=='wrong_shape':v[k]=v[k][:,:-1].contiguous()
    elif mutation=='wrong_dtype':v[k]=v[k].float()
    else:v['unexpected']=torch.ones(1)
    save_file(v,str(path/'model.safetensors'))
    with patch.object(NativeLoader,'read',side_effect=AssertionError('payload forbidden')):
        with pytest.raises(ValueError):GptOssAdapter(path)


def test_export_quantization_budget_and_complete_resume(tmp_path):
    path=tmp_path/'source';a,n,r,s=setup(path);s.close();a.close()
    with patch.object(NativeLoader,'read',side_effect=AssertionError('payload forbidden')):
        with pytest.raises(MemoryError):GptOssAdapter(path,host_budget_bytes=100)
        with pytest.raises(MemoryError):GptOssAdapter(path,device_budget_bytes=100)
        with pytest.raises(ValueError):GptOssAdapter(path,quantization='int4-sym-g128')
        with pytest.raises(ValueError):export_quantized(path,tmp_path/'forbidden','gpt_oss','int4-sym-g128')
    assert not (tmp_path/'forbidden').exists()
    cfg=TrainConfig(str(path),str(path/'tokens.jsonl'),str(tmp_path/'full'),adapter='gpt_oss',steps=2,accumulation=2,weight_decay=.1)
    run(cfg);run(dataclasses.replace(cfg,output=str(tmp_path/'cold')),resume=tmp_path/'full/checkpoint-1')
    for folder,file in [('checkpoint-2','state.safetensors'),('adapter','state.safetensors'),('peft-adapter','adapter_model.safetensors')]:
        x=load_file(str(tmp_path/'full'/folder/file));y=load_file(str(tmp_path/'cold'/folder/file));assert x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x)


def test_head_fits_but_working_set_rejects_before_any_payload(tmp_path):
    path=tmp_path/'source';a,n,r,s=setup(path);s.close();a.close();report=inspect_gpt_oss(path)
    from ram_vram_finetune.adapters.gpt_oss.adapter import working_set
    budget=working_set(report,2)
    assert budget['device_charge']>report['head_bytes']
    with patch.object(NativeLoader,'read',side_effect=AssertionError('payload forbidden')):
        with pytest.raises(MemoryError):GptOssAdapter(path,device_budget_bytes=report['head_bytes']+1)
        with pytest.raises(MemoryError):GptOssAdapter(path,host_budget_bytes=report['host_weight_bytes']+1)


def test_native_whole_right_padding_gradient(tmp_path):
    a,n,refs,s=setup(tmp_path/'source');ids=torch.tensor([[1,2,3,4,5],[2,6,9,0,0]]);labels=torch.tensor([[-100,-100,3,4,5],[-100,6,9,-100,-100]]);valid=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
    x=n.model.embed_tokens(ids).detach().requires_grad_();out=n(inputs_embeds=x,attention_mask=valid,use_cache=False,output_router_logits=False)
    expected=F.cross_entropy(out.logits[:,:-1].float().reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')/5;expected.backward()
    # Public microbatch accepts per-row all-valid IDs; exercise block graph with
    # an independent native batched pad mask, including padded query positions.
    from ram_vram_finetune.training.recompute import streaming_block
    xx=a.embedding[ids].detach().requires_grad_();h=xx;pos,masks=a.positions_and_masks(valid)
    with s.run():
        for b in a.blocks:h=streaming_block(h,pos,masks[b.kind],b.functional,b.kwargs,s.reloader(b),b.adapters,b.index)
        logits=F.linear(a.terminal_norm(h),a.head);ell=F.cross_entropy(logits[:,:-1].float().reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')/5;ell.backward()
    compare(ell,expected);compare(xx.grad,x.grad)
    for key,p in flatten(a).items():compare(p.grad_A,refs[key].A.grad);compare(p.grad_B,refs[key].B.grad)
    s.close();a.close()
