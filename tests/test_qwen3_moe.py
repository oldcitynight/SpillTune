"""Whole independent HF Qwen3-MoE oracle and strict source contracts."""
import dataclasses,json
from unittest.mock import patch
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from ram_vram_finetune.qwen3_moe_fixture import state as fixture_state
from ram_vram_finetune.adapters.qwen3_moe import Qwen3MoeAdapter,inspect_qwen3_moe,hf_classes,moe_feed_forward
from ram_vram_finetune.quantization.storage import WeightLoader,export_quantized
from ram_vram_finetune.training.engine import flatten,microbatch,run
from ram_vram_finetune.finetune.optimizer import LoRAOptimizer
from ram_vram_finetune.memory import SlotSession
from ram_vram_finetune.config import TrainConfig


def setup(path,*,layout='per_expert_2d',dtype=torch.float32,top_k=1,norm=True,tied=True):
    torch.set_num_threads(1);torch.manual_seed(729);path.mkdir()
    config,source=fixture_state(2,layout=layout,experts=3,top_k=top_k,norm_topk=norm,tied=tied)
    source={k:v.to(dtype) for k,v in source.items()};save_file(source,str(path/'model.safetensors'));(path/'config.json').write_text(json.dumps(config.to_dict()))
    rows=[{'input_ids':[1,2,3,4,5],'labels':[-100,-100,3,4,5]},{'input_ids':[3,6,9],'labels':[-100,-100,9]}]
    (path/'tokens.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    adapter=Qwen3MoeAdapter(path)
    # Independent native state construction from canonical source. No adapter weights/mask/rope.
    native_weights={k:v.clone() for k,v in source.items()}
    if layout=='per_expert_2d':
        for i in range(2):
            prefix=f'model.layers.{i}.mlp.experts.';gate=[];up=[];down=[]
            for e in range(3):
                gate.append(native_weights.pop(f'{prefix}{e}.gate_proj.weight'));up.append(native_weights.pop(f'{prefix}{e}.up_proj.weight'));down.append(native_weights.pop(f'{prefix}{e}.down_proj.weight'))
            native_weights[prefix+'gate_up_proj']=torch.cat((torch.stack(gate),torch.stack(up)),dim=1)
            native_weights[prefix+'down_proj']=torch.stack(down)
    if tied:native_weights['lm_head.weight']=native_weights['model.embed_tokens.weight']
    Config,_,Model=hf_classes();cfg=Config(**json.loads((path/'config.json').read_text()));cfg._attn_implementation='eager';cfg._experts_implementation='eager'
    native=Model(cfg).eval().requires_grad_(False);native.load_state_dict(native_weights,strict=True,assign=True)
    if tied:native.tie_weights();assert native.lm_head.weight is native.model.embed_tokens.weight
    assert native.model.rotary_emb.inv_freq.dtype==torch.float32
    refs={}
    class Projection(torch.nn.Module):
        def __init__(self,base,p):
            super().__init__();self.base=base;self.A=torch.nn.Parameter(p.A.detach().clone());self.B=torch.nn.Parameter(p.B.detach().clone());self.scale=p.scale
        def forward(self,x):
            base=self.base(x);return (base.float()+F.linear(F.linear(x.float(),self.A),self.B)*self.scale).to(base.dtype)
    for b in adapter.blocks:
        for target,p in b.adapters.items():
            parent,leaf=target.rsplit('.',1);owner=native.model.layers[b.index].get_submodule(parent);ref=Projection(getattr(owner,leaf),p);setattr(owner,leaf,ref);refs[f'{b.index}:{target}']=ref
    slot=SlotSession(adapter.blocks,'cpu',host_budget_bytes=1024**3,device_budget_bytes=1024**3)
    return adapter,native,refs,slot


def native_loss(native,ids,labels):
    x=native.model.embed_tokens(ids).detach().requires_grad_(True)
    logits=native(inputs_embeds=x,use_cache=False,output_router_logits=False).logits
    loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')
    return loss,x


def compare(a,b,dtype):
    torch.testing.assert_close(a,b,rtol=.03 if dtype==torch.bfloat16 else 3e-5,atol=2e-3 if dtype==torch.bfloat16 else 3e-6)
    relative=float((a-b).detach().float().norm())/max(float(b.detach().float().norm()),1e-12)
    assert relative<(.08 if dtype==torch.bfloat16 else 3e-4)


@pytest.mark.parametrize('layout',['per_expert_2d','packed_3d'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('top_k',[1,2])
@pytest.mark.parametrize('norm',[True,False])
@pytest.mark.parametrize('nonzero_b',[False,True])
def test_whole_native_loss_input_and_every_lora_gradient(tmp_path,layout,dtype,top_k,norm,nonzero_b):
    adapter,native,refs,slot=setup(tmp_path/'model',layout=layout,dtype=dtype,top_k=top_k,norm=norm)
    projections=flatten(adapter)
    if nonzero_b:
        with torch.no_grad():
            for key,p in projections.items():p.B.normal_(0,.03);refs[key].B.copy_(p.B)
    ids=torch.tensor([[1,3,5,7,9]]);labels=torch.tensor([[-100,-100,5,7,9]])
    expected,x=native_loss(native,ids,labels);(expected/3).backward()
    with slot.run():actual,gradient=microbatch(adapter,slot,ids,labels,normalization=3,token_chunk=2)
    assert actual==pytest.approx(float(expected.detach()/3),abs=3e-6,rel=3e-6);compare(gradient,x.grad,dtype)
    assert torch.count_nonzero(gradient[:,:2])>0
    for key,p in projections.items():compare(p.grad_A,refs[key].A.grad,dtype);compare(p.grad_B,refs[key].B.grad,dtype)
    slot.close();adapter.close()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('norm',[False,True])
def test_two_updates_unequal_accumulation_and_optimizer(tmp_path,dtype,norm):
    adapter,native,refs,slot=setup(tmp_path/'model',dtype=dtype,top_k=2,norm=norm)
    with torch.no_grad():
        for key,p in flatten(adapter).items():p.B.normal_(0,.02);refs[key].B.copy_(p.B)
    optimizer=LoRAOptimizer(lr=.001,weight_decay=.1);oracle=torch.optim.AdamW([t for p in refs.values() for t in (p.A,p.B)],lr=.001,weight_decay=.1,foreach=False)
    rows=[(torch.tensor([[1,2,3,4,5]]),torch.tensor([[-100,-100,3,4,5]])),(torch.tensor([[3,6,9]]),torch.tensor([[-100,-100,9]]))]
    frozen={f'{b.index}:{k}':v.clone() for b in adapter.blocks for k,v in b.host_weights.items()}
    for _ in range(2):
        oracle.zero_grad(set_to_none=True);optimizer.zero_grad(flatten(adapter));loss=sum(native_loss(native,*row)[0] for row in rows)/4;loss.backward()
        with slot.run():actual=sum(microbatch(adapter,slot,*row,normalization=4,token_chunk=2,accumulate=i>0)[0] for i,row in enumerate(rows))
        assert actual==pytest.approx(float(loss.detach()),abs=3e-6,rel=3e-6)
        for key,p in flatten(adapter).items():
            compare(p.grad_A,refs[key].A.grad,dtype);compare(p.grad_B,refs[key].B.grad,dtype)
            if dtype==torch.bfloat16:refs[key].A.grad.copy_(p.grad_A);refs[key].B.grad.copy_(p.grad_B)
        optimizer.step(flatten(adapter));oracle.step()
        for key,p in flatten(adapter).items():
            for field in ('A','B'):
                compare(getattr(p,field),getattr(refs[key],field),torch.float32)
                for prefix,name in [('m_','exp_avg'),('v_','exp_avg_sq')]:compare(getattr(p,prefix+field),oracle.state[getattr(refs[key],field)][name],torch.float32)
    assert all(torch.equal(v,frozen[f'{b.index}:{k}']) for b in adapter.blocks for k,v in b.host_weights.items())
    slot.close();adapter.close()


def test_native_router_input_vjp_and_dormant_aux_objective(tmp_path):
    adapter,native,refs,slot=setup(tmp_path/'model',top_k=2,norm=False)
    x=torch.randn(1,5,16,requires_grad=True);weights=adapter.blocks[0].host_weights
    expected=native.model.layers[0].mlp(x);actual=moe_feed_forward(x,weights,num_experts=3,top_k=2,norm_topk_prob=False)
    compare(actual,expected,torch.float32);a,=torch.autograd.grad(actual.square().sum(),x,retain_graph=True);b,=torch.autograd.grad(expected.square().sum(),x);compare(a,b,torch.float32)
    logits=F.linear(x.reshape(-1,16),weights['mlp.gate.weight']);score=F.softmax(logits.float(),-1)
    route,indices=torch.topk(score,2,dim=-1)
    detached_output=native.model.layers[0].mlp.experts(x.reshape(-1,16),indices,route.detach()).reshape_as(x)
    detached_grad,=torch.autograd.grad(detached_output.square().sum(),x,retain_graph=True)
    # A frozen router still contributes d(score)/d(input); this control must differ.
    assert float((b-detached_grad).norm())/max(float(b.norm()),1e-12)>1e-3
    assert torch.autograd.grad(score[:,0].sum(),x)[0].abs().sum()>0
    metadata=adapter.runtime_metadata();assert metadata['objective']=='CE_only' and not metadata['auxiliary_loss_enabled'] and metadata['router_aux_loss_coef_config']==.001
    ids=torch.tensor([[1,2,3,4,5]]);labels=torch.tensor([[-100,-100,3,4,5]])
    out=native(input_ids=ids,labels=labels,use_cache=False,output_router_logits=False)
    assert out.aux_loss is None and out.router_logits is None
    manual=F.cross_entropy(out.logits[:,:-1].float().reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100)
    compare(out.loss,manual,torch.float32)
    slot.close();adapter.close()


@pytest.mark.parametrize('change',[{'output_router_logits':True},{'num_experts_per_tok':4},{'mlp_only_layers':[0]},{'decoder_sparse_step':2},
    {'attention_bias':True},{'sliding_window':2,'use_sliding_window':True},{'norm_topk_prob':1},{'router_aux_loss_coef':-1}])
def test_reject_unsupported_before_payload(tmp_path,change):
    path=tmp_path/'model';adapter,native,refs,slot=setup(path);slot.close();adapter.close()
    raw=json.loads((path/'config.json').read_text());raw.update(change);(path/'config.json').write_text(json.dumps(raw))
    with patch.object(WeightLoader,'read',side_effect=AssertionError('payload read prohibited')):
        with pytest.raises((ValueError,TypeError)):Qwen3MoeAdapter(path)


@pytest.mark.parametrize('mutation',['missing_expert','wrong_expert_shape','mixed_layout','duplicate_tied_head'])
def test_exact_schema_and_budget_alias_identity(tmp_path,mutation):
    path=tmp_path/'model';adapter,native,refs,slot=setup(path);slot.close();adapter.close();weights=load_file(str(path/'model.safetensors'))
    if mutation=='missing_expert':weights.pop('model.layers.0.mlp.experts.2.up_proj.weight')
    elif mutation=='wrong_expert_shape':weights['model.layers.0.mlp.experts.0.gate_proj.weight']=torch.ones(7,16)
    elif mutation=='mixed_layout':weights['model.layers.0.mlp.experts.gate_up_proj']=torch.zeros(3,64,16)
    else:weights['lm_head.weight']=weights['model.embed_tokens.weight'].clone()
    save_file(weights,str(path/'model.safetensors'))
    with patch.object(WeightLoader,'read',side_effect=AssertionError('payload read prohibited')):
        with pytest.raises(ValueError):Qwen3MoeAdapter(path)


def test_source_alias_and_quant_budget_rejections(tmp_path):
    path=tmp_path/'model';adapter,native,refs,slot=setup(path);assert adapter.embedding.data_ptr()==adapter.head.data_ptr();assert not adapter.embedding.requires_grad
    assert adapter.source_layout=='per_expert_2d';slot.close();adapter.close()
    with patch.object(WeightLoader,'read',side_effect=AssertionError('payload read prohibited')):
        with pytest.raises(MemoryError):Qwen3MoeAdapter(path,host_budget_bytes=100)
        with pytest.raises(ValueError,match='unsupported_quantization'):Qwen3MoeAdapter(path,quantization='int6-sym-g128')
        with pytest.raises(ValueError,match='unsupported_quantization'):export_quantized(path,tmp_path/'archive','qwen3_moe','int6-sym-g128')


def test_complete_checkpoint_and_exports_same_process_cpu(tmp_path):
    path=tmp_path/'model';adapter,native,refs,slot=setup(path);slot.close();adapter.close()
    cfg=TrainConfig(str(path),str(path/'tokens.jsonl'),str(tmp_path/'full'),adapter='qwen3_moe',steps=2,accumulation=2,weight_decay=.1)
    run(cfg);run(dataclasses.replace(cfg,output=str(tmp_path/'resumed')),resume=tmp_path/'full/checkpoint-1')
    a=load_file(str(tmp_path/'full/checkpoint-2/state.safetensors'));b=load_file(str(tmp_path/'resumed/checkpoint-2/state.safetensors'));assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    for folder,file in [('adapter','state.safetensors'),('peft-adapter','adapter_model.safetensors')]:
        a=load_file(str(tmp_path/'full'/folder/file));b=load_file(str(tmp_path/'resumed'/folder/file));assert a.keys()==b.keys() and len(a)==12 and all(torch.equal(a[k],b[k]) for k in a)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_untied_independent_head_native_vjp(tmp_path,dtype):
    adapter,native,refs,slot=setup(tmp_path/'model',dtype=dtype,tied=False,top_k=2,norm=False)
    assert not adapter.source_aliases and adapter.embedding.data_ptr()!=adapter.head.data_ptr()
    ids=torch.tensor([[1,3,5,7,9]]);labels=torch.tensor([[-100,-100,5,7,9]])
    loss,x=native_loss(native,ids,labels);(loss/3).backward()
    with slot.run():actual,gradient=microbatch(adapter,slot,ids,labels,normalization=3,token_chunk=2)
    assert actual==pytest.approx(float(loss.detach()/3),abs=3e-6,rel=3e-6);compare(gradient,x.grad,dtype)
    for key,p in flatten(adapter).items():compare(p.grad_A,refs[key].A.grad,dtype);compare(p.grad_B,refs[key].B.grad,dtype)
    slot.close();adapter.close()


def test_prompt_only_expert_influences_later_supervised_loss(tmp_path):
    adapter,native,refs,slot=setup(tmp_path/'model',top_k=1,norm=True)
    with torch.no_grad():
        for key in ('q_proj','k_proj','v_proj','o_proj'):
            adapter.blocks[0].host_weights['self_attn.'+key+'.weight'].zero_()
            module=getattr(native.model.layers[0].self_attn,key)
            (module.base if hasattr(module,'base') else module).weight.zero_()
        for tensor in (adapter.embedding,native.model.embed_tokens.weight):
            tensor[1:3,0]=1.;tensor[3:6,0]=-1.
        for tensor in (adapter.blocks[0].host_weights['mlp.gate.weight'],native.model.layers[0].mlp.gate.weight):
            tensor.zero_();tensor[0,0]=4.;tensor[1,0]=-4.
        adapter.blocks[0].host_weights['mlp.experts.down_proj'][0].mul_(50)
        native.model.layers[0].mlp.experts.down_proj[0].mul_(50)
    routes=[]
    handle=native.model.layers[0].mlp.gate.register_forward_hook(lambda module,args,out:routes.append(out[2].detach().clone()))
    ids=torch.tensor([[1,2,3,4,5]]);labels=torch.tensor([[-100,-100,-100,4,5]])
    loss,x=native_loss(native,ids,labels);(loss/2).backward()
    with slot.run():actual,grad=microbatch(adapter,slot,ids,labels,normalization=2,token_chunk=2)
    assert routes[0].flatten().tolist()==[0,0,1,1,1]
    assert actual==pytest.approx(float(loss.detach()/2),abs=3e-6,rel=3e-6);compare(grad,x.grad,torch.float32)
    assert float(x.grad[:,:2].abs().sum())>1e-6
    # The expert selected solely by unlabeled prompt tokens affects a later
    # attention layer's supervised output; skipping it is not a valid shortcut.
    with torch.no_grad():native.model.layers[0].mlp.experts.down_proj[0].zero_()
    removed,_=native_loss(native,ids,labels)
    assert abs(float(removed.detach()-loss.detach()))>1e-7
    handle.remove();slot.close();adapter.close()
