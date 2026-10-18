"""Behavioral contracts: adapter policy, arbitrary targets and real typed parts."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ram_vram_finetune.adapters.registry import inspect_model, register
from ram_vram_finetune.buffers import RawBufferContract, contract_bindings
from ram_vram_finetune.config import TrainConfig
from ram_vram_finetune.finetune.lora import create_lora_for_block
from ram_vram_finetune.fixture import make_tiny
from ram_vram_finetune.memory import SlotSession, decoded_layout
from ram_vram_finetune.quantization.codecs import FORMATS, DTYPES
from ram_vram_finetune.quantization.storage import export_quantized, require_export_capability
from ram_vram_finetune.training.engine import flatten, microbatch, run
from typed_adapter_fixture import make_source, TypedAdapter, register_fixture, NAME, TARGET


def test_arbitrary_targets_are_explicit_and_aliases_are_instance_scoped():
    targets=['branch_a.readout','branch_b.readout','third.final']
    adapters=create_lora_for_block(0,'full_attention',{t+'.weight':(4,6) for t in targets},targets=targets)
    assert list(adapters)==targets and adapters['final'] is adapters['third.final']
    assert 'readout' not in adapters
    with pytest.raises(KeyError):adapters['readout']
    with pytest.raises(ValueError,match='explicit'):create_lora_for_block(0,'full_attention',{})


@pytest.mark.parametrize('targets',[[],['a','a'],['a..b'],['a/b'],[''], 'module'])
def test_invalid_target_mappings_fail_before_random_allocation(targets):
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError):create_lora_for_block(0,'linear_attention',{},targets=targets)
    assert torch.equal(before,torch.get_rng_state())


@pytest.mark.parametrize('capability',[None,{}, {'version':True,'formats':[],'rejection_reason':''},
    {'version':1,'formats':['int4-sym-g128','int4-sym-g128'],'rejection_reason':''},
    {'version':1,'formats':['native-unknown'],'rejection_reason':''},
    {'version':1,'formats':'int4-sym-g128','rejection_reason':''},
    {'version':1,'formats':[],'rejection_reason':''}])
def test_export_missing_or_invalid_capability_has_no_output_side_effect(tmp_path,monkeypatch,capability):
    import ram_vram_finetune.adapters.registry as registry
    import ram_vram_finetune.quantization.storage as storage
    monkeypatch.setattr(registry,'inspect_model',lambda *_:{'quantized_export':capability})
    monkeypatch.setattr(storage.WeightLoader,'read',lambda *_:pytest.fail('payload read before capability'))
    output=tmp_path/'not-created'/'archive'
    with pytest.raises(ValueError,match='unsupported_quantization'):export_quantized(tmp_path,output,'unknown_future_model','int4-sym-g128')
    assert not output.parent.exists()


@pytest.mark.parametrize('family',['llama','qwen2','qwen35','qwen3','qwen3_moe','deepseek_v3'])
def test_builtin_capability_matrix_and_pre_payload_denial(tmp_path,monkeypatch,family):
    import ram_vram_finetune.quantization.storage as storage
    path=make_tiny(tmp_path/'model',adapter=family,layers=2)
    declaration=inspect_model(path,family)['quantized_export']
    assert json.loads(json.dumps(declaration))==declaration
    allowed=family not in ('qwen3','qwen3_moe')
    assert declaration['formats']==(list(FORMATS) if allowed else [])
    for mode in FORMATS:
        if allowed:require_export_capability({'quantized_export':declaration},mode)
        else:
            monkeypatch.setattr(storage.WeightLoader,'read',lambda *_:pytest.fail('denied payload'))
            with pytest.raises(ValueError,match='unsupported_quantization'):export_quantized(path,tmp_path/'new'/'archive',family,mode)
            assert not (tmp_path/'new').exists()


def test_unknown_registered_inspector_can_explicitly_export_without_common_changes(tmp_path):
    from ram_vram_finetune.adapters.dense import inspect_dense
    path=make_tiny(tmp_path/'model',adapter='llama',layers=1)
    register('arbitrary_export_contract',lambda *_:None,inspector=lambda p:inspect_dense(p,'llama'))
    result=export_quantized(path,tmp_path/'archive','arbitrary_export_contract','int4-sym-g128')
    assert (tmp_path/'archive'/'COMPLETE').is_file()


def block_for(dtype=torch.uint8,contract=True):
    return SimpleNamespace(index=0,storage_specs=None,host_weights={'odd':torch.arange(12).to(dtype).reshape(3,4)},
        buffer_contract=RawBufferContract('arbitrary-decoder','a'*64,128) if contract else None)


@pytest.mark.parametrize('dtype',[torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,torch.bool,torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_typed_parts_preserve_dtype_bytes_alignment_and_lifecycle(dtype):
    b=block_for(dtype);session=SlotSession([b],'cpu',host_budget_bytes=4096,device_budget_bytes=4096)
    with session.run():
        reload=session.reloader(b);view=reload()['odd']
        assert view.dtype==dtype and view.data_ptr()%256==0 and torch.equal(view,b.host_weights['odd'])
        assert session.trace[-1]['decode_workspace_budget_bytes']==128
    with pytest.raises(RuntimeError):reload()
    session.close()
    assert DTYPES=={'bfloat16':torch.bfloat16,'float32':torch.float32}


def test_workspace_combined_budget_and_metadata_mutation_fail_closed():
    b=block_for();capacity=decoded_layout(b)[0]
    with pytest.raises(MemoryError):SlotSession([b],'cpu',host_budget_bytes=4096,device_budget_bytes=capacity+127)
    with pytest.raises(MemoryError):SlotSession([b],'cpu',host_budget_bytes=12+capacity+127,device_budget_bytes=4096)
    s=SlotSession([b],'cpu',host_budget_bytes=4096,device_budget_bytes=4096)
    with s.run():
        b.host_weights['odd']=torch.arange(12,dtype=torch.uint8).reshape(4,3)
        with pytest.raises(ValueError,match='layout changed'):s.load(b)
    s.close()
    with pytest.raises(ValueError,match='explicit'):SlotSession([block_for(contract=False)],'cpu',host_budget_bytes=4096,device_budget_bytes=4096)


@pytest.mark.parametrize('identity,workspace',[('bad',0),('a'*64,-1),('a'*64,True)])
def test_malformed_raw_contract_rejected(identity,workspace):
    with pytest.raises(ValueError):RawBufferContract('opaque',identity,workspace)


def test_new_contract_rejects_unvalidated_transfer_before_cuda_context():
    b=block_for()
    with pytest.raises(ValueError,match='synchronous'):SlotSession([b],'cuda:0',host_budget_bytes=4096,device_budget_bytes=4096,transfer_mode='batched')
    from ram_vram_finetune.prefetch import PrefetchSession
    with pytest.raises(ValueError,match='synchronous'):PrefetchSession([b],'cuda:0',host_budget_bytes=4096,device_budget_bytes=4096)


def test_typed_decode_input_and_every_lora_vjp(tmp_path):
    torch.set_num_threads(1);path=make_source(tmp_path/'model');adapter=TypedAdapter(path)
    p=next(iter(flatten(adapter).values()))
    with torch.no_grad():p.B.fill_(.03)
    ids=torch.tensor([[1,2,3,4]]);labels=torch.tensor([[-100,-100,3,4]])
    hidden=adapter.embedding[ids].detach().requires_grad_(True)
    # Independent algebra rather than the adapter functional or LoRA.forward.
    block=adapter.blocks[0];weight=(block.host_weights['packed'].float()-18)*block.host_weights['scale']
    value=hidden+torch.tanh(F.linear(hidden,weight)+F.linear(F.linear(hidden,p.A),p.B)*(p.alpha/p.rank))
    loss=F.cross_entropy(F.linear(value,adapter.head)[:,:-1].reshape(-1,13),labels[:,1:].reshape(-1),reduction='sum',ignore_index=-100)/2
    expected=torch.autograd.grad(loss,(hidden,p.A,p.B))
    slot=SlotSession(adapter.blocks,'cpu',host_budget_bytes=100000,device_budget_bytes=100000)
    with slot.run():actual,grad=microbatch(adapter,slot,ids,labels,normalization=2,token_chunk=1)
    assert actual==pytest.approx(float(loss.detach()),abs=1e-6)
    for a,b in zip((grad,p.grad_A,p.grad_B),expected):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    assert slot.reloads==2
    slot.close();adapter.close()


@pytest.mark.parametrize('mutation',['identity','workspace','layout'])
def test_typed_accumulation_export_and_new_pid_restore_and_identity_rejection(tmp_path,mutation):
    torch.set_num_threads(1);register_fixture();path=make_source(tmp_path/'model')
    config=TrainConfig(str(path),str(path/'tokens.jsonl'),str(tmp_path/'full'),adapter=NAME,steps=2,accumulation=2,token_chunk=1)
    run(config)
    cold=dataclasses.replace(config,output=str(tmp_path/'cold'))
    request={'config':dataclasses.asdict(cold),'resume':str(tmp_path/'full/checkpoint-1'),'result':str(tmp_path/'child.json')}
    file=tmp_path/'request.json';file.write_text(json.dumps(request))
    child=subprocess.run([sys.executable,str(Path(__file__).with_name('typed_adapter_fixture.py')),str(file)],capture_output=True,text=True,timeout=60)
    assert child.returncode==0,child.stdout+child.stderr
    info=json.loads((tmp_path/'child.json').read_text());assert info['pid']!=os.getpid()
    import ram_vram_finetune
    assert info['package_file']==ram_vram_finetune.__file__
    for name in ['checkpoint-2/state.safetensors','adapter/state.safetensors','peft-adapter/adapter_model.safetensors']:
        a=load_file(str(tmp_path/'full'/name));b=load_file(str(tmp_path/'cold'/name));assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    metadata=json.loads((tmp_path/'full/checkpoint-1/metadata.json').read_text())
    assert metadata['binding']['typed_buffers'][0]['identity']==hashlib.sha256(b'fixture-decode-v1').hexdigest()
    exported=load_file(str(tmp_path/'cold/adapter/state.safetensors'))
    assert len(exported)==2
    for folder in ('full','cold'):
        meta=json.loads((tmp_path/folder/'adapter/metadata.json').read_text())
        companion=json.loads((tmp_path/folder/'peft-adapter/rvft_typed_buffers.json').read_text())
        assert meta['typed_buffers']==companion['typed_buffers']==metadata['binding']['typed_buffers']
        assert companion['requires_matching_interpretation_and_base'] is True
    patch=json.loads((path/'contract.json').read_text())
    if mutation=='identity':patch['identity']='b'*64
    elif mutation=='workspace':patch['workspace']+=1
    else:patch['packed_shape']=[4,9]
    (path/'contract.json').write_text(json.dumps(patch))
    with pytest.raises(ValueError,match='binding'):run(dataclasses.replace(config,output=str(tmp_path/'rejected')),resume=tmp_path/'full/checkpoint-1')


def test_tied_export_is_still_denied_before_payload_and_publication(tmp_path,monkeypatch):
    import ram_vram_finetune.quantization.storage as storage
    from safetensors.torch import save_file
    path=make_tiny(tmp_path/'model',adapter='qwen35',layers=2)
    config=json.loads((path/'config.json').read_text());config['tie_word_embeddings']=True;(path/'config.json').write_text(json.dumps(config))
    values=load_file(str(path/'model.safetensors'));values.pop('lm_head.weight');save_file(values,str(path/'model.safetensors'))
    monkeypatch.setattr(storage.WeightLoader,'read',lambda *_:pytest.fail('tied payload'))
    assert inspect_model(path,'qwen35')['quantized_export']['formats']==[]
    with pytest.raises(ValueError,match='tied'):export_quantized(path,tmp_path/'new'/'archive','qwen35','int4-sym-g128')
    assert not (tmp_path/'new').exists()


def test_contract_declaration_is_deep_snapshot_and_exception_releases_run(tmp_path):
    path=make_source(tmp_path/'model');adapter=TypedAdapter(path);block=adapter.blocks[0]
    before={k:t.clone() for k,t in block.host_weights.items()}
    slot=SlotSession(adapter.blocks,'cpu',host_budget_bytes=100000,device_budget_bytes=100000)
    binding=contract_bindings(adapter.blocks);binding[0]['parts'][0]['shape'][0]=123
    assert slot.buffer_metadata[id(block)]['parts'][0]['shape'][0]==6
    with pytest.raises(RuntimeError,match='injected'):
        with slot.run():
            stale=slot.reloader(block);stale()
            raise RuntimeError('injected decoder failure')
    with slot.run():
        with pytest.raises(RuntimeError,match='owner'):stale()
        assert slot.load(block)
    assert all(torch.equal(before[k],v) for k,v in block.host_weights.items())
    slot.close();adapter.close()


@pytest.mark.parametrize('limited',['host','device'])
def test_exact_typed_workspace_budget_success_and_one_byte_under_failure(limited):
    b=block_for();capacity=decoded_layout(b)[0]
    device=capacity+255+128
    host=12+device
    session=SlotSession([b],'cpu',host_budget_bytes=host,device_budget_bytes=device)
    session.close()
    caps={'host_budget_bytes':host,'device_budget_bytes':device}
    caps[limited+'_budget_bytes']-=1
    with pytest.raises(MemoryError):SlotSession([b],'cpu',**caps)
    # With the identical physical buffers, changing only workspace must change admission.
    b.buffer_contract=RawBufferContract('arbitrary-decoder','a'*64,129)
    with pytest.raises(MemoryError):SlotSession([b],'cpu',host_budget_bytes=host,device_budget_bytes=device)
