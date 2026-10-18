"""Real native HF DeepSeek-V3 references; no adapter masks/rotary shared."""
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

from ram_vram_finetune.adapters.deepseek import DeepseekV3Adapter,hf_classes,inspect_deepseek
import ram_vram_finetune.adapters.deepseek as ds
from ram_vram_finetune.deepseek_fixture import state as fixture_state
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
    make_tiny(path,adapter='deepseek_v3',layers=2)
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
@pytest.mark.parametrize('mode',MODES)
def test_native_two_updates_router_recompute_quantization(tmp_path,layout,mode,record_property):
    r=numeric_case(tmp_path/'model',mode,layout)
    for k in ['decoder_host_bytes','decoded_slot_bytes']:record_property(k,r[k])
    record_property('max_gradient_error',max(x['max_gradient_abs_error'] for x in r['steps']))


@pytest.mark.parametrize('layout',['packed','individual'])
def test_direct_q_split_rotary_default_fp32(tmp_path,layout):
    numeric_case(tmp_path/'model','none',layout,dtype=torch.float32,q_rank=None,rope='default',interleave=False)


@pytest.mark.parametrize('layout',['packed','individual'])
@pytest.mark.parametrize('mode',FORMATS)
def test_direct_q_split_rotary_default_quantized(tmp_path,layout,mode):
    numeric_case(tmp_path/'model',mode,layout,q_rank=None,rope='default',interleave=False)


def cold_child_code():
    return '''import json,sys,random
from pathlib import Path
import torch
from safetensors.torch import load_file
import ram_vram_finetune as package
from ram_vram_finetune.config import TrainConfig
from ram_vram_finetune.training.engine import run
from ram_vram_finetune.checkpoint_io import state
assert str(Path(package.__file__).resolve())==sys.argv[4]
config=TrainConfig(**json.loads(Path(sys.argv[1]).read_text()))
original=state.load_checkpoint
def restore(target,projections,optimizer,binding):
 expected=load_file(str(Path(target)/'state.safetensors'));metadata=json.loads((Path(target)/'metadata.json').read_text())
 with torch.no_grad():
  for p in projections.values():
   for f in state.FIELDS:getattr(p,f).fill_(.125)
 random.seed(991);torch.manual_seed(991);optimizer.step_count=99;optimizer.lr=.123
 assert not torch.equal(torch.get_rng_state(),expected['rng/cpu'])
 cursor=original(target,projections,optimizer,binding)
 assert cursor==metadata['cursor'] and optimizer.state_dict()==metadata['optimizer']
 assert all(torch.equal(getattr(p,f).detach().cpu(),expected[k+'/'+f]) for k,p in projections.items() for f in state.FIELDS)
 assert torch.equal(torch.get_rng_state(),expected['rng/cpu']) and json.loads(json.dumps(random.getstate()))==metadata['python_rng']
 if config.device=='cuda:0':assert torch.equal(torch.cuda.get_rng_state(0),expected['rng/cuda'])
 return cursor
state.load_checkpoint=restore
if sys.argv[3]!='none':
 from safetensors import safe_open
 try:
  with safe_open(str(Path(sys.argv[3])/'model.safetensors'),framework='pt'):pass
 except (PermissionError,FileNotFoundError):pass
 else:raise AssertionError('source remained available')
r=run(config,resume=sys.argv[2]);assert r['status']=='completed' and len(r['metrics'])==1
print(json.dumps({'status':'PASS','package_file':package.__file__,'restored_poisoned_state':True}))
'''


def cold_case(path,mode,layout,device='cpu',isolated_full=False):
    path.mkdir();source=fixture(path/'source',layout);data=path/'tokens.jsonl';data.write_bytes((source/'tokens.jsonl').read_bytes())
    compressed=path/'compressed'
    if mode!='none':export_quantized(source,compressed,'deepseek_v3',mode)
    config=TrainConfig(str(source),str(data),str(path/'full'),adapter='deepseek_v3',device=device,steps=2,accumulation=2,quantization=mode)
    env=dict(os.environ);env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if env.get('PYTHONPATH'):env['PYTHONPATH']=os.pathsep.join(str(Path(p or '.').resolve()) for p in env['PYTHONPATH'].split(os.pathsep))
    if isolated_full:
        cfg=path/'full.json';cfg.write_text(json.dumps(dataclasses.asdict(config)))
        code="from pathlib import Path;import json,sys;from ram_vram_finetune.config import TrainConfig;from ram_vram_finetune.training.engine import run;assert run(TrainConfig(**json.loads(Path(sys.argv[1]).read_text())))['status']=='completed'"
        completed=subprocess.run([sys.executable,'-c',code,str(cfg)],cwd='/tmp',env=env,capture_output=True,text=True,timeout=45)
        (path/'full.log').write_text(completed.stdout+completed.stderr);assert completed.returncode==0,completed.stderr
    else:run(config)
    cfg=path/'resume.json';cfg.write_text(json.dumps(dataclasses.asdict(dataclasses.replace(config,model=str(compressed) if mode!='none' else str(source),output=str(path/'resumed')))))
    original_mode=source.stat().st_mode&0o777
    try:
        if mode!='none':source.chmod(0);assert not os.access(source,os.R_OK|os.X_OK)
        package_file=str(Path(__import__('ram_vram_finetune').__file__).resolve())
        completed=subprocess.run([sys.executable,'-c',cold_child_code(),str(cfg),str(path/'full/checkpoint-1'),str(source) if mode!='none' else 'none',package_file],cwd='/tmp',env=env,capture_output=True,text=True,timeout=45)
    finally:source.chmod(original_mode)
    (path/'resume.log').write_text(completed.stdout+completed.stderr);assert completed.returncode==0,completed.stderr
    a=load_file(str(path/'full/checkpoint-2/state.safetensors'));b=load_file(str(path/'resumed/checkpoint-2/state.safetensors'))
    assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    am=json.loads((path/'full/checkpoint-2/metadata.json').read_text());bm=json.loads((path/'resumed/checkpoint-2/metadata.json').read_text());assert am==bm
    for name,file in [('adapter','state.safetensors'),('peft-adapter','adapter_model.safetensors')]:
        a1=load_file(str(path/'full'/name/file));b1=load_file(str(path/'resumed'/name/file));assert a1.keys()==b1.keys() and all(torch.equal(a1[k],b1[k]) for k in a1)
    return {'status':'PASS','format':mode,'layout':layout,'device':device,'exact_checkpoint_tensors':len(a),'metadata_exact':True,
        'poisoned_state_restored_in_new_process':True,'bf16_source_unavailable':mode!='none',
        'full':json.loads((path/'full/result.json').read_text()),'resumed':json.loads((path/'resumed/result.json').read_text()),
        'resource_scope':'separate full and resume child processes' if isolated_full else 'CPU full in test process; resume fresh child'}


@pytest.mark.parametrize('layout',['packed','individual'])
@pytest.mark.parametrize('mode',MODES)
def test_cold_resume_all_state_exports_without_original_quantized_source(tmp_path,mode,layout):
    cold_case(tmp_path/'job',mode,layout)


@pytest.mark.parametrize('change',[
    {'model_type':'deepseek_v2'},{'model_type':'deepseek_v32'},{'model_type':'deepseek_v4'},{'model_type':'qwen2'},
    {'quantization_config':{'quant_method':'fp8'}},{'num_nextn_predict_layers':1},{'aux_loss_alpha':.01},
    {'scoring_func':'softmax'},{'topk_method':'greedy'},{'n_group':3},{'topk_group':3},{'num_experts_per_tok':3},
    {'tie_word_embeddings':True},{'attention_dropout':.1},{'attention_bias':True},{'moe_layer_freq':2},
    {'ep_size':2},{'index_topk':2},{'rope_parameters':{'rope_type':'dynamic','rope_theta':10000.,'factor':2.}},
])
def test_unsupported_architecture_and_training_rejected(tmp_path,change):
    p=fixture(tmp_path/'model');raw=json.loads((p/'config.json').read_text());raw.update(change);(p/'config.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError):inspect_deepseek(p)


def test_layout_shape_finite_router_and_before_payload_budgets(tmp_path,monkeypatch):
    p=fixture(tmp_path/'model');weights=load_file(str(p/'model.safetensors'));key='model.layers.1.mlp.gate.weight'
    weights[key][0,0]=float('nan');save_file(weights,str(p/'model.safetensors'))
    with pytest.raises(ValueError,match='nonfinite router'):DeepseekV3Adapter(p)
    weights[key][0,0]=0;weights['model.layers.1.mlp.experts.0.gate_proj.weight']=torch.zeros(16,16,dtype=torch.bfloat16)
    save_file(weights,str(p/'model.safetensors'))
    with pytest.raises(ValueError,match='inventory'):inspect_deepseek(p)
    weights.pop('model.layers.1.mlp.experts.0.gate_proj.weight');save_file(weights,str(p/'model.safetensors'))
    monkeypatch.setattr(ds.WeightLoader,'read',lambda *a:pytest.fail('budget must reject before payload'))
    with pytest.raises(MemoryError,match='before loading'):DeepseekV3Adapter(p,device_budget_bytes=1)


def test_mask_rotary_and_router_edge_contracts(tmp_path):
    p=fixture(tmp_path/'model');a=DeepseekV3Adapter(p);native,_,_,_=references(a,p)
    ids=torch.tensor([[1,2,3,4,5]]);valid=torch.tensor([[1,1,1,0,0]],dtype=torch.bool)
    s=SlotSession(a.blocks,'cpu',host_budget_bytes=1024**3,device_budget_bytes=1024**3)
    x=F.embedding(ids,a.embedding).detach().requires_grad_(True);pos,mask=a.positions_and_masks(valid)
    with s.run():actual=a.terminal_norm(forward(a,s,x,pos,mask))
    expected=native.model(inputs_embeds=x,attention_mask=valid,use_cache=False).last_hidden_state
    torch.testing.assert_close(actual,expected,rtol=.02,atol=.002)
    with pytest.raises(ValueError,match='right padding'):a.positions_and_masks(torch.tensor([[0,1]],dtype=torch.bool))
    # Tie selection reproducible on one backend; no cross-backend tie promise.
    values=[ds.route(torch.ones(3,16),torch.zeros(4,16),torch.zeros(4),groups=2,group_k=1,top_k=2,normalize=False,scale=1.7) for _ in range(3)]
    assert all(torch.equal(v[2],values[0][2]) for v in values) and torch.all(values[0][1]==.85)
    s.close();a.close()


@pytest.mark.parametrize('truncate',[True,False])
def test_yarn_real_rotary_dimension_and_sparse_long_positions(truncate):
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding
    Config,_,_=hf_classes()
    c=Config(hidden_size=128,num_attention_heads=2,num_key_value_heads=2,qk_rope_head_dim=64,
        max_position_embeddings=163840,rope_parameters={'rope_type':'yarn','rope_theta':10000.,'factor':40.,
        'original_max_position_embeddings':4096,'beta_fast':32.,'beta_slow':1.,'mscale':1.,'mscale_all_dim':1.,'truncate':truncate})
    inverse,amplitude=ds.rope_parameters(c,'cpu');native=DeepseekV3RotaryEmbedding(c)
    assert torch.equal(inverse,native.inv_freq) and amplitude==native.attention_scaling
    # Sparse positions exercise long context without an O(sequence^2) mask.
    positions=torch.tensor([[0,512,4096,163839]])
    phase=(inverse[None,:,None]@positions[:,None].float()).transpose(1,2);phase=torch.cat((phase,phase),-1)
    expected=native(torch.empty(1,4,128,dtype=torch.bfloat16),positions)
    assert torch.equal((phase.cos()*amplitude).to(torch.bfloat16),expected[0])
    assert torch.equal((phase.sin()*amplitude).to(torch.bfloat16),expected[1])


@pytest.mark.parametrize('normalize',[True,False])
def test_router_bias_changes_selection_not_selected_probability(normalize):
    import math
    x=torch.ones(1,1);w=torch.tensor([[-2.],[-1.],[.1],[.2]])
    kwargs={'groups':2,'group_k':1,'top_k':2,'normalize':normalize,'scale':1.7}
    _,plain,ids0=ds.route(x,w,torch.zeros(4),**kwargs)
    _,selected,ids=ds.route(x,w,torch.tensor([2.,2.,0.,0.]),**kwargs)
    assert set(ids0.flatten().tolist())=={2,3} and set(ids.flatten().tolist())=={0,1}
    probabilities=[1/(1+math.exp(-float(w[i,0]))) for i in ids[0]]
    expected=torch.tensor([p/(sum(probabilities) if normalize else 1)*1.7 for p in probabilities])
    torch.testing.assert_close(selected[0],expected,rtol=1e-6,atol=1e-7)
