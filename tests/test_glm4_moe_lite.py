"""Real native HF GLM4MoELite references; no adapter masks/rotary shared."""
import copy
import dataclasses
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file

from ram_vram_finetune.adapters.glm4_moe_lite import Glm4MoeLiteAdapter as DeepseekV3Adapter,hf_classes,inspect_glm4_moe_lite as inspect_deepseek
import ram_vram_finetune.adapters.deepseek as ds
from ram_vram_finetune.glm_fixture import state as fixture_state
from ram_vram_finetune.fixture import make_tiny
from ram_vram_finetune.config import TrainConfig
from ram_vram_finetune.training.engine import flatten,microbatch,run,forward
from ram_vram_finetune.finetune.optimizer import LoRAOptimizer
from ram_vram_finetune.memory import SlotSession
from ram_vram_finetune.quantization.codecs import FORMATS
from ram_vram_finetune.quantization.storage import export_quantized

MODES=('none',*FORMATS)
independent_decode=runpy.run_path(str(Path(__file__).with_name('test_quantization.py')))['independent_decode']


def fixture(path,layout='packed',dtype=torch.bfloat16,**options):
    make_tiny(path,adapter='glm4_moe_lite',layers=2)
    config,state=fixture_state(2,layout=layout,dtype=dtype,**options)
    save_file(state,str(path/'model.safetensors'));(path/'config.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n')
    return path


def native_state(state,config):
    """Independent checkpoint layout mapping to HF5.13 packed experts."""
    result={k:v.clone() for k,v in state.items()}
    for i in range(config.first_k_dense_replace,config.num_hidden_layers):
        prefix=f'model.layers.{i}.mlp.experts.'
        if prefix+'gate_up_proj' in result:continue
        result[prefix+'gate_up_proj']=torch.stack([torch.cat((result.pop(prefix+f'{e}.gate_proj.weight'),
            result.pop(prefix+f'{e}.up_proj.weight')),0) for e in range(config.n_routed_experts)])
        result[prefix+'down_proj']=torch.stack([result.pop(prefix+f'{e}.down_proj.weight') for e in range(config.n_routed_experts)])
    return result


def references(adapter,path):
    raw=load_file(str(path/'model.safetensors'));mode=adapter.storage_metadata
    baseline={k:v.to(torch.float32 if k.endswith(('.mlp.gate.weight','.mlp.gate.e_score_correction_bias')) else torch.bfloat16) for k,v in raw.items()} if mode else raw
    decoded={k:v.clone() for k,v in baseline.items()};errors=[]
    for block in adapter.blocks:
        if block.storage_specs is None:
            weights={k:v.clone() for k,v in block.host_weights.items()}
        else:
            weights={k:independent_decode(item['spec'],{part:block.host_weights[key] for part,key in item['parts'].items()}) for k,item in block.storage_specs.items()}
        for name,value in weights.items():
            key=f'model.layers.{block.index}.{name}';delta=value.float()-baseline[key].float()
            errors.append(float(delta.norm()/baseline[key].float().norm().clamp_min(1e-20)));decoded[key]=value
    _,_,Model=hf_classes()
    def model(state):
        m=Model(copy.deepcopy(adapter.config)).eval();m.load_state_dict(native_state(state,adapter.config),strict=True,assign=True)
        return m.requires_grad_(False).to(adapter.device)
    native=model(decoded);original=model(baseline);refs={}
    class Projection(torch.nn.Module):
        def __init__(self,base,p):
            super().__init__();self.base=base;self.A=torch.nn.Parameter(p.A.detach().clone());self.B=torch.nn.Parameter(p.B.detach().clone());self.scale=p.scale
        def forward(self,x):
            y=self.base(x);return (y.float()+F.linear(F.linear(x.float(),self.A),self.B)*self.scale).to(y.dtype)
    for b in adapter.blocks:
        for name,p in b.adapters.items():
            parent,leaf=name.rsplit('.',1);module=native.model.layers[b.index].get_submodule(parent)
            wrapped=Projection(getattr(module,leaf),p);setattr(module,leaf,wrapped);refs[f'{b.index}:{name}']=wrapped
    return native,original,refs,max(errors)


def native_loss(model,ids,labels):
    device=next(model.parameters()).device
    x=model.model.embed_tokens(ids.to(device)).detach().requires_grad_(True)
    logits=model(inputs_embeds=x,use_cache=False).logits
    loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,32),labels[:,1:].to(device).reshape(-1),ignore_index=-100,reduction='sum')
    return loss,x,logits


def numeric_case(path,mode,layout,device='cpu',dtype=torch.bfloat16,**options):
    torch.set_num_threads(1);fixture(path,layout,dtype,**options)
    adapter=DeepseekV3Adapter(path,device=device,quantization=mode)
    native,original,refs,weight_error=references(adapter,path);projections=flatten(adapter)
    slot=SlotSession(adapter.blocks,device,host_budget_bytes=1024**3,device_budget_bytes=1024**3)
    frozen={f'{b.index}:{k}':t.clone() for b in adapter.blocks for k,t in b.host_weights.items()}
    rows=[(torch.tensor([[1,2,3,4,5]]),torch.tensor([[-100,-100,3,4,5]])),
          (torch.tensor([[3,6,9]]),torch.tensor([[-100,-100,9]]))]
    qloss,_,qlogits=native_loss(native,*rows[0]);bloss,_,blogits=native_loss(original,*rows[0])
    lossy={'loss_abs':abs(float((qloss-bloss).detach()/3)),
        'logit_max_abs':float((qlogits-blogits).abs().max().detach()),'weight_max_relative_l2':weight_error}
    if layout=='individual':
        with torch.no_grad():
            for k,p in projections.items():p.B.normal_(0,.01);refs[k].B.copy_(p.B)
    # Both grouped routed and shared paths must change the actual function.
    probe=torch.arange(1,25).remainder(31).reshape(1,-1)
    router=native.model.layers[1].mlp.gate
    h=native.model.embed_tokens(probe.to(adapter.device));_,_,probe_ids=router(h)
    assert probe_ids.unique().numel()>=3,'fixture must hit multiple routed groups/experts'
    baseline_logits=native(input_ids=probe.to(adapter.device),use_cache=False).logits
    hook=native.model.layers[1].mlp.shared_experts.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
    assert not torch.equal(baseline_logits,native(input_ids=probe.to(adapter.device),use_cache=False).logits);hook.remove()
    produced=[];expected=[];original_route=ds.route
    def observed(*a,**kw):
        result=original_route(*a,**kw);produced.append((result[2].detach().cpu().tolist(),result[1].detach().cpu().clone()));return result
    ds.route=observed
    hook=router.register_forward_hook(lambda m,a,o:expected.append((o[2].detach().cpu().tolist(),o[1].detach().cpu().clone())))
    opt=LoRAOptimizer(lr=.001,weight_decay=.1)
    expected_opt=torch.optim.AdamW([t for r in refs.values() for t in (r.A,r.B)],lr=.001,weight_decay=.1,foreach=False)
    steps=[]
    try:
        if device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            expected.clear();produced.clear();expected_opt.zero_grad(set_to_none=True)
            results=[native_loss(native,*row) for row in rows];reference=sum(x[0] for x in results)/4;reference.backward()
            opt.zero_grad(projections)
            cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(0).clone() if device.startswith('cuda') else None
            with slot.run():
                actual=[microbatch(adapter,slot,*row,normalization=4,token_chunk=2,accumulate=i>0) for i,row in enumerate(rows)]
                loss=sum(x[0] for x in actual)
                assert abs(loss-float(reference.detach()))<2e-6
                pairs=[(x[1],y[1].grad) for x,y in zip(actual,results)]
                pairs += [(getattr(p,'grad_'+f),getattr(refs[k],f).grad) for k,p in projections.items() for f in ('A','B')]
                worst_grad=max(float((a-b).abs().max()) for a,b in pairs)
                for a,b in pairs:torch.testing.assert_close(a,b,rtol=.02 if dtype==torch.bfloat16 else 2e-5,atol=5e-4 if dtype==torch.bfloat16 else 2e-6)
                gradient_metrics=[]
                labels=['input_'+str(i) for i in range(len(actual))]+[k+'/'+f for k in projections for f in ('A','B')]
                for label,(a,b) in zip(labels,pairs):
                    norm=float(b.float().norm());relative=float((a-b).float().norm())/max(norm,1e-12)
                    assert relative<(.05 if dtype==torch.bfloat16 else .0001)
                    gradient_metrics.append({'name':label,'max_abs_error':float((a-b).abs().max()),'relative_l2_error':relative,
                        'reference_norm':norm,'actual_norm':float(a.float().norm()),'actual_nonzero':int(torch.count_nonzero(a))})
                assert len(produced)==4 and len(expected)==2
                for i in range(2):
                    assert produced[2*i][0]==produced[2*i+1][0]==expected[i][0]
                    torch.testing.assert_close(produced[2*i][1],expected[i][1],rtol=1e-5,atol=1e-7)
                    assert torch.equal(produced[2*i][1],produced[2*i+1][1])
                assert torch.equal(cpu_rng,torch.get_rng_state())
                if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state(0))
                # Separate oracles: HF checks gradients with BF16 tolerance;
                # AdamW must receive IDENTICAL measured gradients to isolate
                # optimizer correctness from low-precision backward rounding.
                if dtype==torch.bfloat16:
                    for k,p in projections.items():
                        refs[k].A.grad=p.grad_A.detach().clone();refs[k].B.grad=p.grad_B.detach().clone()
                opt.step(projections)
            expected_opt.step();pairs=[]
            for k,p in projections.items():
                for f in ('A','B'):
                    r=getattr(refs[k],f);pairs.append((getattr(p,f),r))
                    pairs.extend([(getattr(p,prefix+f),expected_opt.state[r][field]) for prefix,field in [('m_','exp_avg'),('v_','exp_avg_sq')]])
            for a,b in pairs:torch.testing.assert_close(a,b,rtol=0 if dtype==torch.bfloat16 else 2e-5,atol=0 if dtype==torch.bfloat16 else 2e-6)
            steps.append({'step':step+1,'loss_abs_error':abs(loss-float(reference.detach())),
                'max_gradient_abs_error':worst_grad,'gradient_metrics':gradient_metrics,'max_update_moment_abs_error':max(float((a-b).abs().max().detach()) for a,b in pairs)})
        assert all(torch.equal(t,frozen[f'{b.index}:{k}']) for b in adapter.blocks for k,t in b.host_weights.items())
        assert [x['block'] for x in slot.trace]==[0,1,1,0]*4
        result={'format':mode,'layout':layout,'device':device,'dtype':str(dtype),'options':options,'steps':steps,'lossy_vs_bf16':lossy,
            'native_router_and_forward_recompute_ids_weights_equal':True,'no_rng_consumed_by_recomputation':True,
            'optimizer_oracle':'identical streamed gradients after independent HF gradient assertions; not independent HF BF16 trajectory' if dtype==torch.bfloat16 else 'independent HF FP32 gradient/update trajectory',
            'routed_probe_unique_experts':int(probe_ids.unique().numel()),'shared_path_has_effect':True,'base_frozen':True,
            'decoder_host_bytes':slot.host_bytes,'pinned_arena_bytes':slot.pinned_capacity_bytes,'decoded_slot_bytes':slot.capacity,
            'upload_slot_bytes':slot.upload_capacity,'actual_h2d_bytes':slot.actual_h2d_bytes,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0,
            'resource_scope':'native/original/streamed graphs jointly resident; correctness only','trace':slot.trace,'runtime':adapter.runtime_metadata()}
        return result
    finally:ds.route=original_route;hook.remove();slot.close();adapter.close()



@pytest.mark.parametrize('layout',['packed','individual'])
@pytest.mark.parametrize('mode,dtype',[('none',torch.float32),('none',torch.bfloat16),('int4-sym-g128',torch.bfloat16)])
def test_native_glm_two_updates(tmp_path,layout,mode,dtype):
    result=numeric_case(tmp_path/'model',mode,layout,dtype=dtype)
    assert len(result['steps'])==2

@pytest.mark.parametrize('field,value',[("model_type","deepseek_v3"),("num_nextn_predict_layers",2),("attention_bias",True),("tie_word_embeddings",True),("aux_loss_alpha",.1),("output_router_logits",True),("q_lora_rank",None),("mlp_layer_types",["sparse","dense"]),("rope_interleave",1),("pretraining_tp",2),("partial_rotary_factor",.5)])
def test_reject_before_payload(tmp_path,field,value,monkeypatch):
    path=fixture(tmp_path/'model','individual');raw=json.loads((path/'config.json').read_text());raw[field]=value;(path/'config.json').write_text(json.dumps(raw))
    from ram_vram_finetune.adapters.glm4_moe_lite import GlmLoader
    monkeypatch.setattr(GlmLoader,'read',lambda *a:pytest.fail('payload touched'))
    with pytest.raises((ValueError,TypeError)):DeepseekV3Adapter(path)

@pytest.mark.parametrize('mtp',[False,True])
def test_mtp_identity_and_explicit_exclusion(tmp_path,mtp):
    path=fixture(tmp_path/'model','individual',mtp=mtp);report=inspect_deepseek(path)
    adapter=DeepseekV3Adapter(path,quantization='int4-sym-g128')
    assert adapter.source_read_keys==report['required'];assert len(adapter.blocks)==2
    assert adapter.runtime_metadata()['excluded_mtp_key_count']==len(report['excluded_mtp'])
    before=adapter.identity;adapter.close()
    if mtp:
        state=load_file(str(path/'model.safetensors'));key='model.layers.2.eh_proj.weight';state[key]=state[key]+.01;save_file(state,str(path/'model.safetensors'))
        other=DeepseekV3Adapter(path,quantization='int4-sym-g128');assert other.identity!=before;other.close()

@pytest.mark.parametrize('which',['host','device'])
def test_budget_before_payload(tmp_path,which,monkeypatch):
    path=fixture(tmp_path/'model','individual');from ram_vram_finetune.adapters.glm4_moe_lite import GlmLoader
    monkeypatch.setattr(GlmLoader,'read',lambda *a:pytest.fail('payload touched'))
    kw={which+'_budget_bytes':1}
    with pytest.raises(MemoryError):DeepseekV3Adapter(path,**kw)

def test_mtp_missing_extra_shape_rejected(tmp_path):
    path=fixture(tmp_path/'model','individual',mtp=True);state=load_file(str(path/'model.safetensors'));state['model.layers.2.eh_proj.weight']=torch.zeros(16,16);save_file(state,str(path/'model.safetensors'))
    with pytest.raises(ValueError,match='inventory/shape'):inspect_deepseek(path)

def test_quantized_export_capability_rejects_before_load(tmp_path):
    path=fixture(tmp_path/'model','individual');
    with pytest.raises(ValueError,match='MTP base archive'):export_quantized(path,tmp_path/'export','glm4_moe_lite','int4-sym-g128')

def test_right_padding_and_reject_holes(tmp_path):
    path=fixture(tmp_path/'model','individual');a=DeepseekV3Adapter(path)
    valid=torch.tensor([[True,True,False]]);pos,masks=a.positions_and_masks(valid);assert masks['mla_moe'].shape==(1,1,3,3)
    native,_,_=references(a,path)[:3];ids=torch.tensor([[1,2,3]]);x=F.embedding(ids,a.embedding).detach().requires_grad_();session=SlotSession(a.blocks,'cpu',host_budget_bytes=1024**3,device_budget_bytes=1024**3)
    with session.run():actual=a.terminal_norm(forward(a,session,x,pos,masks))
    expected=native.model(inputs_embeds=x,attention_mask=valid,use_cache=False).last_hidden_state
    torch.testing.assert_close(actual,expected,atol=.002,rtol=.02);session.close()
    with pytest.raises(ValueError):a.positions_and_masks(torch.tensor([[True,False,True]]))
    a.close()

def test_router_probability_gradient_is_required(tmp_path,monkeypatch):
    path=fixture(tmp_path/'model','individual',dtype=torch.float32);a=DeepseekV3Adapter(path);b=a.blocks[1]
    pos,masks=a.positions_and_masks(torch.ones(1,8,dtype=torch.bool));x=torch.randn(1,8,16,requires_grad=True)
    weights={k:v for k,v in b.host_weights.items()};out=b.functional(x,pos,masks[b.kind],**b.kwargs,**weights);probe=torch.randn_like(out);good=torch.autograd.grad(out,x,probe)[0]
    original=ds.route
    def detached(*args,**kwargs):
        logits,scores,indices=original(*args,**kwargs);return logits,scores.detach(),indices
    monkeypatch.setattr(ds,'route',detached);xx=x.detach().clone().requires_grad_();badout=b.functional(xx,pos,masks[b.kind],**b.kwargs,**weights);bad=torch.autograd.grad(badout,xx,probe)[0]
    assert torch.equal(out,badout);assert float((good-bad).abs().max())>1e-7;a.close()

@pytest.mark.parametrize('mode',['none','int4-sym-g128'])
def test_three_layers_prompt_vjp(tmp_path,mode):
    path=tmp_path/'model';path.mkdir();c,s=fixture_state(3,layout='individual',dtype=torch.bfloat16);save_file(s,str(path/'model.safetensors'));(path/'config.json').write_text(json.dumps(c.to_dict()))
    a=DeepseekV3Adapter(path,quantization=mode);native,_,refs,_=references(a,path)
    with torch.no_grad():
        for k,p in flatten(a).items():p.B.normal_(0,.01);refs[k].B.copy_(p.B)
    ids=torch.tensor([[1,2,3,4,5,6]]);labels=torch.tensor([[-100,-100,-100,4,5,6]])
    expected,x,_=native_loss(native,ids,labels);expected.backward();session=SlotSession(a.blocks,'cpu',host_budget_bytes=1024**3,device_budget_bytes=1024**3)
    with session.run():loss,g=microbatch(a,session,ids,labels,normalization=1,token_chunk=2)
    assert abs(loss-float(expected.detach()))<2e-6;assert torch.count_nonzero(g[:,0])>0
    torch.testing.assert_close(g,x.grad,atol=.002,rtol=.03)
    for k,p in flatten(a).items():
        for f in ('A','B'):torch.testing.assert_close(getattr(p,'grad_'+f),getattr(refs[k],f).grad,atol=.002,rtol=.03)
    session.close();a.close()

def test_mtp_only_change_rejects_old_checkpoint(tmp_path):
    path=fixture(tmp_path/'model','individual',mtp=True);cfg=TrainConfig(str(path),str(path/'tokens.jsonl'),str(tmp_path/'run'),adapter='glm4_moe_lite',steps=1,quantization='int4-sym-g128');run(cfg)
    state=load_file(str(path/'model.safetensors'));state['model.layers.2.eh_proj.weight']+=.01;save_file(state,str(path/'model.safetensors'))
    with pytest.raises(ValueError,match='binding|identity'):run(dataclasses.replace(cfg,output=str(tmp_path/'resumed'),steps=2),resume=tmp_path/'run/checkpoint-1')

def test_prompt_only_experts_before_later_attention(tmp_path,monkeypatch):
    path=tmp_path/'model';path.mkdir();c,s=fixture_state(3,layout='individual',dtype=torch.float32);save_file(s,str(path/'model.safetensors'));(path/'config.json').write_text(json.dumps(c.to_dict()))
    a=DeepseekV3Adapter(path);native,_,refs,_=references(a,path)
    # Controlled route intervention isolates experts2/3 to prompt token0 in
    # both independent graphs. Their base-weight gradients demonstrate that
    # a later attention layer carries future supervised loss through them.
    def selection(logits):
        indices=torch.tensor([[2,3]]+[[0,1]]*(logits.shape[0]-1),device=logits.device)
        scores=logits.sigmoid().gather(1,indices);return scores/(scores.sum(-1,keepdim=True)+1e-20)*c.routed_scaling_factor,indices
    handle=native.model.layers[1].mlp.gate.register_forward_hook(lambda m,args,out:(out[0],*selection(out[0])))
    original=ds.route;calls=[]
    def controlled(*args,**kwargs):
        logits,score,index=original(*args,**kwargs)
        # two sparse layers called once each forward, then reversed backward
        # re-evaluation; identify layer1 by the frozen router pointer.
        if torch.equal(args[1],a.blocks[1].host_weights['mlp.gate.weight']):score,index=selection(logits)
        return logits,score,index
    monkeypatch.setattr(ds,'route',controlled)
    native.model.layers[1].mlp.experts.gate_up_proj.requires_grad_(True)
    ids=torch.tensor([[1,2,3,4,5,6]]);labels=torch.tensor([[-100,-100,-100,4,5,6]]);expected,x,_=native_loss(native,ids,labels);expected.backward()
    assert torch.count_nonzero(native.model.layers[1].mlp.experts.gate_up_proj.grad[3])>0
    session=SlotSession(a.blocks,'cpu',host_budget_bytes=1024**3,device_budget_bytes=1024**3)
    with session.run():loss,grad=microbatch(a,session,ids,labels,normalization=1,token_chunk=2)
    assert abs(loss-float(expected.detach()))<2e-6;torch.testing.assert_close(grad,x.grad,atol=4e-6,rtol=4e-5)
    for k,p in flatten(a).items():
        for f in ('A','B'):torch.testing.assert_close(getattr(p,'grad_'+f),getattr(refs[k],f).grad,atol=4e-6,rtol=4e-5)
    handle.remove();session.close();a.close()

def test_exact_combined_budget_thresholds(tmp_path,monkeypatch):
    from ram_vram_finetune.adapters.glm4_moe_lite import working_set,GlmLoader
    path=fixture(tmp_path/'model','individual');report=inspect_deepseek(path);b=working_set(report,'int4-sym-g128',2,cuda=False)
    a=DeepseekV3Adapter(path,quantization='int4-sym-g128',host_budget_bytes=b['host_before_payload'],device_budget_bytes=b['device_before_payload']);a.close()
    monkeypatch.setattr(GlmLoader,'read',lambda *a:pytest.fail('payload touched'))
    for h,d in [(b['host_before_payload']-1,b['device_before_payload']),(b['host_before_payload'],b['device_before_payload']-1)]:
        with pytest.raises(MemoryError):DeepseekV3Adapter(path,quantization='int4-sym-g128',host_budget_bytes=h,device_budget_bytes=d)

@pytest.mark.parametrize('kwargs',[{'rank':0},{'rank':-1},{'rank':True},{'rank':1.5},{'alpha':float('nan')},{'alpha':float('inf')},{'alpha':float('-inf')}])
def test_invalid_lora_rejects_before_payload(tmp_path,kwargs,monkeypatch):
    path=fixture(tmp_path/'model','individual');from ram_vram_finetune.adapters.glm4_moe_lite import GlmLoader
    monkeypatch.setattr(GlmLoader,'read',lambda *a:pytest.fail('payload touched'))
    with pytest.raises(ValueError,match='LoRA'):DeepseekV3Adapter(path,**kwargs)
