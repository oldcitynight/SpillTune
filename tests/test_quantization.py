"""Independent scalar bit decoder + native HF graphs, never codec-vs-itself."""
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file

from ram_vram_finetune.adapters.registry import resolve
from ram_vram_finetune.adapters.dense import hf_classes
from ram_vram_finetune.adapters.qwen35.adapter import SUFFIXES
from ram_vram_finetune.config import TrainConfig
from ram_vram_finetune.fixture import make_tiny
from ram_vram_finetune.finetune.optimizer import LoRAOptimizer
from ram_vram_finetune.memory import SlotSession
from ram_vram_finetune.training.engine import flatten,microbatch,run
from ram_vram_finetune.quantization.codecs import FORMATS,encode as encode_tensor,materialize,validate_spec,Record
from ram_vram_finetune.quantization.storage import export_quantized,MARKER,file_hash


def encode(value,key,mode):
    # Codec receives no architecture/key policy; these codec tests select a
    # matrix explicitly. Production selection belongs to adapter reports.
    return encode_tensor(value,mode)


def scalar_fp8(code,mode):
    exponent_bits,mantissa_bits,bias=(4,3,7) if 'e4m3fn' in mode else (5,2,15)
    sign=-1 if code&128 else 1
    exp=(code>>mantissa_bits)&((1<<exponent_bits)-1);man=code&((1<<mantissa_bits)-1)
    if exp==0:return sign*math.ldexp(man/(1<<mantissa_bits),1-bias)
    if ('e4m3fn' in mode and exp==15 and man==7) or ('e5m2' in mode and exp==31):return float('nan')
    return sign*math.ldexp(1+man/(1<<mantissa_bits),exp-bias)


def independent_decode(spec,parts):
    if spec['codec']=='raw':return parts['value'].detach().cpu().clone()
    raw=bytes(parts['payload'].detach().cpu().tolist());mode=spec['codec'];scales=parts['scales'].detach().cpu().tolist()
    values=[]
    for i in range(spec['numel']):
        if mode=='int4-sym-g128':value=((raw[i//2]>>(4*(i%2)))&15)-8
        elif mode=='int6-sym-g128':
            word=int.from_bytes(raw[(i//4)*3:(i//4)*3+3],'little')
            code=(word>>(6*(i%4)))&63;value=code-32
        elif mode=='int8-sym-g128':value=raw[i] if raw[i]<128 else raw[i]-256
        else:value=scalar_fp8(raw[i],mode)
        # Match the declared FP32 product before the final BF16 rounding.
        values.append(value*scales[i//128])
    return torch.tensor(values,dtype=torch.float32).to(torch.bfloat16).reshape(spec['shape'])


@pytest.mark.parametrize('mode',FORMATS)
@pytest.mark.parametrize('length',[128,129,257])
def test_codec_independent_bits_tails_zeros_and_freeze(mode,length):
    torch.set_num_threads(1);torch.manual_seed(43)
    value=torch.randn(1,length)*.03;value[:,:128]=0
    record=encode(value,'model.layers.0.mlp.up_proj.weight',mode)
    validate_spec(record.spec,record.parts)
    expected=independent_decode(record.spec,record.parts)
    assert torch.equal(materialize(record),expected)
    assert not any(t.requires_grad for t in record.parts.values())
    assert record.parts['payload'].numel()==math.ceil(length/128)*({'int4-sym-g128':64,'int6-sym-g128':96}.get(mode,128))
    assert record.parts['scales'][0]==1


@pytest.mark.parametrize('mode,limit',[('int4-sym-g128',7),('int8-sym-g128',127),('int6-sym-g128',31)])
def test_integer_ties_even_known_vector(mode,limit):
    value=torch.zeros(1,128);value[0,:6]=torch.tensor([-limit,-1.5,-.5,.5,1.5,limit])
    record=encode(value,'model.layers.0.mlp.up_proj.weight',mode)
    assert record.parts['scales'][0]==1
    assert independent_decode(record.spec,record.parts).flatten()[:6].tolist()==[-limit,-2,0,0,2,limit]
    if 'int4' in mode:assert record.parts['payload'][:3].tolist()==[0x61,0x88,0xfa]
    if 'int6' in mode:
        assert bytes(record.parts['payload'][:3].tolist())==(1|(30<<6)|(32<<12)|(32<<18)).to_bytes(3,'little')


@pytest.mark.parametrize('mode,limit,one,midpoints',[
    ('fp8-e4m3fn-g128',448,0x38,(1.0625,1.1875)),
    ('fp8-e5m2-g128',57344,0x3c,(1.125,1.375)),
])
def test_fp8_encoder_known_bits_and_ties_even(mode,limit,one,midpoints):
    value=torch.zeros(1,128)
    value[0,:9]=torch.tensor([limit,1,-1,2,-2,0,*midpoints,-midpoints[0]])
    record=encode_tensor(value,mode)
    assert record.parts['scales'].item()==1
    maximum=0x7e if 'e4m3fn' in mode else 0x7b
    assert record.parts['payload'][:9].tolist()==[maximum,one,one|128,0x40,0xc0,0,one,one+2,one|128]


@pytest.mark.parametrize('mode',FORMATS)
def test_codec_crosses_real_chunk_boundary(mode):
    from ram_vram_finetune.quantization.codecs import CHUNK_GROUPS
    torch.set_num_threads(1)
    length=CHUNK_GROUPS*128+129
    value=(torch.arange(length,dtype=torch.float32).remainder(29)-14).reshape(1,-1)/32
    record=encode_tensor(value,mode)
    validate_spec(record.spec,record.parts)
    assert torch.equal(materialize(record),independent_decode(record.spec,record.parts))
    # Re-encode the final group of chunk0 plus both groups of chunk1 as a
    # single small call. This also catches encoder offset/chunk write errors.
    local=encode_tensor(value[:,(CHUNK_GROUPS-1)*128:].contiguous(),mode)
    stride={'int4-sym-g128':64,'int6-sym-g128':96}.get(mode,128)
    assert torch.equal(record.parts['payload'][-3*stride:],local.parts['payload'])
    assert torch.equal(record.parts['scales'][-3:],local.parts['scales'])


@pytest.mark.parametrize('mode',FORMATS)
def test_reserved_nonfinite_input_scale_and_payload_rejected(mode):
    x=torch.ones(1,128);x[0,0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):encode(x,'model.layers.0.mlp.up_proj.weight',mode)
    record=encode(torch.ones(1,128),'model.layers.0.mlp.up_proj.weight',mode)
    record.parts['scales'][0]=0
    with pytest.raises(ValueError,match='scales'):validate_spec(record.spec,record.parts)
    record.parts['scales'][0]=1
    record.parts['payload'][0]=0 if mode in ('int4-sym-g128','int6-sym-g128') else (128 if mode=='int8-sym-g128' else 127)
    with pytest.raises(ValueError,match='reserved|nonfinite'):validate_spec(record.spec,record.parts)


@pytest.mark.parametrize('mode',FORMATS)
def test_bf16_overflow_and_nonzero_padding_rejected(mode):
    with pytest.raises(ValueError,match='BF16'):
        encode(torch.full((1,128),3.4e38),'model.layers.0.mlp.up_proj.weight',mode)
    record=encode(torch.ones(1,128),'model.layers.0.mlp.up_proj.weight',mode)
    record.parts['scales'].fill_(1e38)
    with pytest.raises(ValueError,match='BF16'):validate_spec(record.spec,record.parts)
    record=encode(torch.ones(1,129),'model.layers.0.mlp.up_proj.weight',mode)
    if mode=='int4-sym-g128':record.parts['payload'][-1]^=1
    elif mode=='int6-sym-g128':record.parts['payload'][-1]|=4
    else:record.parts['payload'][-1]=1
    with pytest.raises(ValueError,match='padding'):validate_spec(record.spec,record.parts)


def build_reference(adapter,path):
    """Native whole HF graph + independently decoded frozen tensors."""
    raw=load_file(str(path/'model.safetensors'))
    # BF16 compute baseline with explicit Qwen delta-state F32 exceptions.
    baseline={k:v.to(torch.float32 if k.endswith(('.dt_bias','.A_log')) else torch.bfloat16) for k,v in raw.items()}
    state={k:v.clone() for k,v in baseline.items()};errors=[]
    qwen=adapter.runtime_metadata()['architecture']=='qwen3_5_text'
    for block in adapter.blocks:
        for name,item in block.storage_specs.items():
            parts={p:block.host_weights[k] for p,k in item['parts'].items()}
            decoded=independent_decode(item['spec'],parts)
            logical=next(k for k,v in SUFFIXES.items() if v==name) if qwen else name
            key=('model.language_model.' if qwen else 'model.')+f'layers.{block.index}.'+logical
            if item['spec']['codec']!='raw':
                diff=decoded.float()-baseline[key].float()
                errors.append((float(diff.abs().max()),float(diff.norm()/baseline[key].float().norm().clamp_min(1e-20))))
            state[key]=decoded
    if qwen:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        Model=Qwen3_5TextModel;state_key=lambda k:k.removeprefix('model.language_model.')
        def native(weights):
            model=Model(adapter.config).eval()
            model.load_state_dict({state_key(k):v for k,v in weights.items() if k.startswith('model.language_model.')},strict=True,assign=True)
            model.register_buffer('_test_head',weights['lm_head.weight'].clone())
            return model.requires_grad_(False)
    else:
        _,_,Model=hf_classes(adapter.family)
        def native(weights):
            model=Model(adapter.config).eval();model.load_state_dict(weights,strict=True,assign=True)
            return model.requires_grad_(False)
    model=native(state);original=native(baseline);refs={}
    class Projection(torch.nn.Module):
        def __init__(self,base,p):
            super().__init__();self.base=base;self.A=torch.nn.Parameter(p.A.detach().clone());self.B=torch.nn.Parameter(p.B.detach().clone());self.scale=p.scale
        def forward(self,x):
            base=self.base(x);return (base.float()+F.linear(F.linear(x.float(),self.A),self.B)*self.scale).to(base.dtype)
    for block in adapter.blocks:
        for name,p in block.adapters.items():
            parent,leaf=name.rsplit('.',1)
            layers=model.layers if qwen else model.model.layers
            module=layers[block.index].get_submodule(parent)
            projection=Projection(getattr(module,leaf),p);setattr(module,leaf,projection);refs[f'{block.index}:{name}']=projection
    def loss(native,ids,labels):
        device=next(native.parameters()).device;ids=ids.to(device);labels=labels.to(device)
        embed=native.embed_tokens if qwen else native.model.embed_tokens
        x=embed(ids).detach().requires_grad_(True)
        if qwen:logits=F.linear(native(inputs_embeds=x,use_cache=False).last_hidden_state,native._test_head)
        else:logits=native(inputs_embeds=x,use_cache=False).logits
        value=F.cross_entropy(logits[:,:-1].float().reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')
        return value,x,logits
    return model,original,refs,loss,errors


@pytest.mark.parametrize('family',['llama','qwen2','qwen35'])
@pytest.mark.parametrize('mode',FORMATS)
def test_quantized_native_two_updates_all_gradients_and_real_storage(tmp_path,family,mode,record_property):
    torch.set_num_threads(1);path=make_tiny(tmp_path/'model',adapter=family,layers=2)
    adapter=resolve(family)(path,quantization=mode)
    slot=SlotSession(adapter.blocks,'cpu',host_budget_bytes=1024**3,device_budget_bytes=1024**3)
    native,original,refs,loss_fn,weight_errors=build_reference(adapter,path)
    projections=flatten(adapter)
    optimizer=LoRAOptimizer(lr=.001,weight_decay=.1)
    expected=torch.optim.AdamW([t for r in refs.values() for t in (r.A,r.B)],lr=.001,weight_decay=.1,foreach=False)
    frozen={f'{b.index}/{k}':v.clone() for b in adapter.blocks for k,v in b.host_weights.items()}
    rows=[(torch.tensor([[1,2,3,4,5]]),torch.tensor([[-100,-100,3,4,5]])),
          (torch.tensor([[3,6,9]]),torch.tensor([[-100,-100,9]]))]
    quant_loss=loss_fn(native,*rows[0])[0]/3;source_loss=loss_fn(original,*rows[0])[0]/3
    record_property('base_loss_abs_error',float((quant_loss-source_loss).abs().detach()))
    record_property('weight_max_abs_error',max(x[0] for x in weight_errors))
    record_property('weight_max_relative_l2_error',max(x[1] for x in weight_errors))
    worst_grad=worst_update=0.
    for step in range(2):
        expected.zero_grad(set_to_none=True)
        results=[loss_fn(native,*row) for row in rows]
        native_loss=sum(x[0] for x in results)/4;native_loss.backward()
        optimizer.zero_grad(projections)
        with slot.run():
            micros=[microbatch(adapter,slot,*row,normalization=4,token_chunk=1,accumulate=i>0) for i,row in enumerate(rows)]
            assert sum(x[0] for x in micros)==pytest.approx(float(native_loss.detach()),abs=2e-6)
            pairs=[(x[1],y[1].grad) for x,y in zip(micros,results)]
            pairs += [(getattr(p,'grad_'+field),getattr(refs[k],field).grad) for k,p in projections.items() for field in ('A','B')]
            for actual,reference in pairs:
                torch.testing.assert_close(actual,reference,rtol=.06,atol=.003)
                worst_grad=max(worst_grad,float((actual-reference).abs().max()))
            optimizer.step(projections)
        expected.step()
        for key,p in projections.items():
            for field in ('A','B'):
                actual=getattr(p,field);reference=getattr(refs[key],field)
                torch.testing.assert_close(actual,reference,rtol=.02,atol=5e-5)
                worst_update=max(worst_update,float((actual-reference).abs().max().detach()))
                for prefix,torchfield in [('m_','exp_avg'),('v_','exp_avg_sq')]:
                    torch.testing.assert_close(getattr(p,prefix+field),expected.state[reference][torchfield],rtol=.06,atol=.003)
    assert all(torch.equal(v,frozen[f'{b.index}/{k}']) for b in adapter.blocks for k,v in b.host_weights.items())
    assert all(t['stored_bytes']<t['decoded_bytes'] for t in slot.trace)
    assert all(not t.requires_grad for b in adapter.blocks for t in b.host_weights.values())
    assert all(b.host_weights[item['parts']['payload']].dtype==torch.uint8 for b in adapter.blocks for item in b.storage_specs.values() if item['spec']['codec']!='raw')
    record_property('max_abs_gradient_error',worst_grad);record_property('max_abs_update_error',worst_update)
    record_property('decoder_stored_bytes',slot.host_bytes);record_property('decoded_slot_bytes',slot.capacity)
    slot.close();adapter.close()


@pytest.mark.parametrize('mode',FORMATS)
def test_new_process_cold_archive_resume_without_bf16_source(tmp_path,mode):
    torch.set_num_threads(1);source=make_tiny(tmp_path/'source',adapter='llama',layers=2)
    archive=tmp_path/'compressed';export_quantized(source,archive,'llama',mode)
    dataset=tmp_path/'tokens.jsonl';dataset.write_bytes((source/'tokens.jsonl').read_bytes())
    config=TrainConfig(str(source),str(dataset),str(tmp_path/'full'),adapter='llama',steps=3,accumulation=2,quantization=mode)
    full=run(config)
    run(dataclasses.replace(config,output=str(tmp_path/'cancel')),cancel_after_step=1)
    hidden=tmp_path/'unavailable-bf16';source.rename(hidden)
    cold=dataclasses.replace(config,model=str(archive),output=str(tmp_path/'cold'))
    cfg=tmp_path/'cold.json';cfg.write_text(json.dumps(dataclasses.asdict(cold)))
    script='''import sys,json
from pathlib import Path
blocked=[Path(x).resolve() for x in sys.argv[1:3]]
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  p=Path(args[0]).resolve()
  if any(p==root or p.is_relative_to(root) for root in blocked):raise AssertionError('BF16 source access forbidden')
sys.addaudithook(guard)
from ram_vram_finetune.config import TrainConfig
from ram_vram_finetune.training.engine import run
import ram_vram_finetune as package
from safetensors import safe_open
assert str(Path(package.__file__).resolve())==sys.argv[5], 'cold process imported a different package'
try:
 with safe_open(str(blocked[1]/'model.safetensors'),framework='pt'):pass
except (PermissionError,FileNotFoundError) as exc:
 print('native_source_unavailable:',type(exc).__name__,str(exc))
else:raise AssertionError('BF16 source remained readable by native safetensors')
config=TrainConfig(**json.loads(Path(sys.argv[3]).read_text()))
r=run(config,resume=sys.argv[4]);assert r['status']=='completed'
print(json.dumps({'package_file':package.__file__,'package_version':package.__version__,'status':r['status']}))
'''
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if env.get('PYTHONPATH'):env['PYTHONPATH']=os.pathsep.join(str(Path(p or '.').resolve()) for p in env['PYTHONPATH'].split(os.pathsep))
    expected_package=str(Path(__import__('ram_vram_finetune').__file__).resolve())
    original_mode=hidden.stat().st_mode & 0o777
    assert (hidden/'model.safetensors').is_file()
    try:
        hidden.chmod(0) # only this generated tiny fixture; native readers cannot traverse it
        assert not os.access(hidden,os.R_OK|os.X_OK)
        completed=subprocess.run([sys.executable,'-c',script,str(source),str(hidden),str(cfg),str(tmp_path/'cancel/checkpoint-1'),expected_package],cwd='/tmp',env=env,capture_output=True,text=True,timeout=40)
    finally:hidden.chmod(original_mode)
    assert completed.returncode==0,completed.stderr
    left=load_file(str(tmp_path/'full/checkpoint-3/state.safetensors'));right=load_file(str(tmp_path/'cold/checkpoint-3/state.safetensors'))
    assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
    assert full['metrics'][-1]['loss']==json.loads((tmp_path/'cold/result.json').read_text())['metrics'][-1]['loss']
    sidecar=json.loads((tmp_path/'cold/peft-adapter/rvft_quantized_base.json').read_text())
    assert sidecar['requires_matching_quantized_base'] and sidecar['storage']['format']==mode


def test_quantized_archive_corruption_format_and_shape_rejected(tmp_path):
    source=make_tiny(tmp_path/'source',adapter='qwen2',layers=1);archive=tmp_path/'q'
    export_quantized(source,archive,'qwen2','int6-sym-g128')
    with pytest.raises(ValueError,match='explicitly select'):resolve('qwen2')(archive)
    with pytest.raises(ValueError,match='explicitly select'):resolve('qwen2')(archive,quantization='int8-sym-g128')
    meta=json.loads((archive/MARKER).read_text());file=archive/next(iter(meta['files']))
    file.write_bytes(file.read_bytes()+b'x')
    with pytest.raises(ValueError,match='hash mismatch'):resolve('qwen2')(archive,quantization='int6-sym-g128')
    for mode in ('Q6_K','Q8_0','FP8','nf4'):
        with pytest.raises(ValueError,match='unsupported_quantization'):
            TrainConfig(str(source),'data','output',quantization=mode)
    with pytest.raises(ValueError,match='native'):TrainConfig(str(source),'data','output',quantization='int6-sym-g128',quantized_compute='fp8-native')


def test_direct_pinned_metadata_and_mapped_boundary_lifetime_without_cuda(tmp_path):
    import gc
    from unittest.mock import patch
    from ram_vram_finetune.adapters.qwen35.adapter import inspect_model
    from ram_vram_finetune.quantization.storage import WeightLoader
    source=make_tiny(tmp_path/'source',adapter='qwen35',layers=2)
    raw=load_file(str(source/'model.safetensors'))
    save_file({k:v.to(torch.float32 if k.endswith(('.dt_bias','.A_log')) else torch.bfloat16) for k,v in raw.items()},str(source/'model.safetensors'))
    report=inspect_model(source);keys=[k for k in report['required'] if '.layers.' in k]
    before=file_hash(source/'model.safetensors')
    with pytest.raises(MemoryError,match='direct pinned'):
        WeightLoader(source,report,'int4-sym-g128',pin_keys=keys,host_budget_bytes=1)
    loader=WeightLoader(source,report,'int4-sym-g128',pin_keys=keys,host_budget_bytes=1024**3)
    assert not loader.pinned_owners and loader.pinned_capacities
    with patch.object(torch.Tensor,'clone',side_effect=AssertionError('unexpected anonymous clone')):
        embedding=loader.read('model.language_model.embed_tokens.weight').parts['value']
    # Loader and safe_open handle have gone; Tensor retains the native mapping.
    del loader;gc.collect()
    expected=load_file(str(source/'model.safetensors'))['model.language_model.embed_tokens.weight'].to(torch.bfloat16)
    assert torch.equal(embedding,expected)
    assert torch.isfinite(F.embedding(torch.tensor([[1,2]]),embedding)).all()
    del embedding;gc.collect();assert file_hash(source/'model.safetensors')==before
