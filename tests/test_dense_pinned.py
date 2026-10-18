"""Dense direct-Q4 resource gates, byte identity and mapped/pinned ownership on CPU."""
import gc
import hashlib
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from ram_vram_finetune.adapters.dense import _new_loader,inspect_dense
from ram_vram_finetune.fixture import make_tiny
from ram_vram_finetune.quantization.storage import WeightLoader
from safetensors.torch import load_file,save_file


def real_shape_report():
    entries={};matrices=[]
    def add(key,shape):entries[key]={'shape':shape,'dtype':'BF16'}
    add('model.embed_tokens.weight',[152064,5120]);add('lm_head.weight',[152064,5120]);add('model.norm.weight',[5120])
    for i in range(64):
        prefix=f'model.layers.{i}.'
        for name,shape in {'self_attn.q_proj.weight':[5120,5120],'self_attn.k_proj.weight':[1024,5120],
                           'self_attn.v_proj.weight':[1024,5120],'self_attn.o_proj.weight':[5120,5120],
                           'mlp.gate_proj.weight':[27648,5120],'mlp.up_proj.weight':[27648,5120],
                           'mlp.down_proj.weight':[5120,27648]}.items():
            add(prefix+name,shape);matrices.append(prefix+name)
        for name,n in {'self_attn.q_proj.bias':5120,'self_attn.k_proj.bias':1024,'self_attn.v_proj.bias':1024,
                       'input_layernorm.weight':5120,'post_attention_layernorm.weight':5120}.items():add(prefix+name,[n])
    return {'text_config':{},'required':sorted(entries),'entries':entries,'quantization_matrix_keys':matrices,'quantization_fp32_keys':[]}


def test_real_shape_plan_and_combined_budget_gate_without_payload_or_cuda(tmp_path):
    report=real_shape_report();loader,direct=_new_loader(tmp_path/'missing',report,'int4-sym-g128','cuda:0',20*1024**3)
    assert direct and loader.mapped_source and len(loader.pinned_capacities)==31
    assert sum(loader.pinned_capacities)==16642998272 and not loader.pinned_owners
    old=WeightLoader(tmp_path/'missing',report,'int4-sym-g128',pin_keys=[k for k in report['required'] if k.startswith('model.layers.')],host_budget_bytes=30*1024**3)
    assert sum(old.pinned_capacities)==25769803776
    # Logical host bytes fit this allowance, but actual arenas plus embedding do not.
    with pytest.raises(MemoryError,match='actual pinned capacity'):
        _new_loader(tmp_path/'missing',report,'int4-sym-g128','cuda:0',18200133632-1)
    assert not (tmp_path/'missing').exists()


@pytest.mark.parametrize('device,mode,direct',[
    ('cuda:0','int4-sym-g128',True),('cpu','int4-sym-g128',False),
    ('cuda:0','int6-sym-g128',False),('cpu','int6-sym-g128',False),
    ('cuda:0','int8-sym-g128',False),('cuda:0','fp8-e4m3fn-g128',False),
    ('cuda:0','fp8-e5m2-g128',False),('cuda:0','none',False),('cpu','none',False)])
def test_direct_dense_policy_has_no_format_or_device_fallback(tmp_path,device,mode,direct):
    loader,selected=_new_loader(tmp_path/'absent',real_shape_report(),mode,device,200*1024**3)
    assert selected==direct and loader.mapped_source==direct
    assert bool(loader.pinned_capacities)==direct and not loader.pinned_owners


@pytest.mark.parametrize('limit',[0,-1,True,1.5,3,512*1024**2+1])
def test_invalid_direct_arena_limit_fails_before_allocation(tmp_path,limit):
    report=real_shape_report()
    with pytest.raises(ValueError,match='positive power of two'):
        WeightLoader(tmp_path,report,'int4-sym-g128',pin_keys=['model.layers.0.mlp.down_proj.weight'],host_budget_bytes=30*1024**3,pinned_arena_limit=limit)


@pytest.mark.parametrize("family",["llama","qwen2"])
@pytest.mark.parametrize("archive",[False,True])
def test_direct_dense_bytes_identity_and_retained_view_lifetime_cpu_emulation(tmp_path,family,archive):
    root=make_tiny(tmp_path/'model',adapter=family,layers=2);file=root/'model.safetensors'
    raw=load_file(str(file));save_file({k:v.to(torch.bfloat16) for k,v in raw.items()},str(file));del raw
    original=hashlib.sha256(file.read_bytes()).hexdigest()
    if archive:
        from ram_vram_finetune.quantization.storage import export_quantized
        root=Path(export_quantized(root,tmp_path/'archive',family,'int4-sym-g128')['output'])
    report=inspect_dense(root,family)
    reference=WeightLoader(root,report,'int4-sym-g128');expected={k:reference.read(k) for k in report['required']};identity=reference.finish()
    loader,direct=_new_loader(root,report,'int4-sym-g128','cuda:0',1024**3);assert direct
    actual={};allocations=[];empty=torch.empty
    def cpu_empty(*args,**kwargs):
        if kwargs.pop('pin_memory',False):allocations.append(args[0])
        return empty(*args,**kwargs)
    # This is an ownership/identity emulation, not a real CUDA pinned allocator test.
    with patch.object(torch,'empty',cpu_empty),patch.object(torch.Tensor,'clone',side_effect=AssertionError('source clone in mapped path')):
        for key in report['required']:actual[key]=loader.read(key)
    assert loader.finish()==identity and loader.source_identity==reference.source_identity and loader.storage_identity==reference.storage_identity
    assert sum(allocations)==sum(loader.pinned_capacities)>0
    for key in actual:
        assert actual[key].spec==expected[key].spec
        assert all(torch.equal(t,expected[key].parts[part]) for part,t in actual[key].parts.items())
    del loader,reference;gc.collect()
    assert all(torch.equal(actual[k].parts[part],t) for k,record in expected.items() for part,t in record.parts.items())
    assert hashlib.sha256(file.read_bytes()).hexdigest()==original


def test_packed_copy_exception_releases_python_owners_and_keeps_source(tmp_path):
    import weakref
    root=make_tiny(tmp_path/'model',adapter='qwen2',layers=2);file=root/'model.safetensors'
    original=hashlib.sha256(file.read_bytes()).hexdigest();report=inspect_dense(root,'qwen2')
    references=[];storages=set();empty=torch.empty;copy=torch.Tensor.copy_
    def allocator(*args,**kwargs):
        pinned=kwargs.pop('pin_memory',False);value=empty(*args,**kwargs)
        if pinned:references.append(weakref.ref(value));storages.add(value.untyped_storage().data_ptr())
        return value
    def copier(self,*args,**kwargs):
        if self.untyped_storage().data_ptr() in storages:raise RuntimeError('injected packed copy failure')
        return copy(self,*args,**kwargs)
    def attempt():
        loader,_=_new_loader(root,report,'int4-sym-g128','cuda:0',1024**3)
        for key in report['required']:loader.read(key)
    caught=False
    try:
        with patch.object(torch,'empty',allocator),patch.object(torch.Tensor,'copy_',copier):attempt()
    except RuntimeError as error:
        assert 'injected packed copy' in str(error);caught=True
    gc.collect();assert caught and references and all(ref() is None for ref in references)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==original
