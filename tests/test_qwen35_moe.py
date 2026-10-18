"""Independent native whole-text VJP, update, inventory and rejection checks."""
import json,dataclasses
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from ram_vram_finetune.qwen35_moe_fixture import make_tiny
from ram_vram_finetune.adapters.qwen35_moe import Qwen35MoeAdapter,inspect_qwen35_moe,hf_classes
from ram_vram_finetune.training.engine import flatten,microbatch,run
from ram_vram_finetune.memory import SlotSession
from ram_vram_finetune.finetune.optimizer import LoRAOptimizer
from ram_vram_finetune.config import TrainConfig


def independent_decode(block):
    if block.storage_specs is None:return {k:v.clone() for k,v in block.host_weights.items()}
    out={}
    for k,s in block.storage_specs.items():
        spec=s['spec'];parts={p:block.host_weights[n] for p,n in s['parts'].items()}
        if spec['codec']=='raw':out[k]=parts['value'].clone();continue
        assert spec['codec']=='int4-sym-g128';raw=parts['payload'].numpy();codes=np.empty(raw.size*2,np.float32);codes[0::2]=(raw&15).astype(np.float32)-8;codes[1::2]=(raw>>4).astype(np.float32)-8
        vals=(codes.reshape(-1,128)*parts['scales'].numpy()[:,None]).reshape(-1)[:spec['numel']].copy();out[k]=torch.from_numpy(vals).bfloat16().reshape(spec['shape'])
    return out


def native(adapter):
    Config,h=hf_classes();cfg=Config(**adapter.config.to_dict());cfg._attn_implementation='eager';cfg._experts_implementation='eager'
    model=h.Qwen3_5MoeForCausalLM(cfg).eval().requires_grad_(False).bfloat16();state={'model.embed_tokens.weight':adapter.embedding.clone(),'model.norm.weight':adapter.norm.clone(),'lm_head.weight':adapter.head.clone()}
    for b in adapter.blocks:state.update({f'model.layers.{b.index}.'+k:v for k,v in independent_decode(b).items()})
    model.load_state_dict(state,strict=True,assign=True);model.model.rotary_emb=h.Qwen3_5MoeTextRotaryEmbedding(cfg);assert model.model.rotary_emb.inv_freq.dtype==torch.float32;refs={}
    class Projection(torch.nn.Module):
        def __init__(self,base,p):
            super().__init__();self.base=base;self.A=torch.nn.Parameter(p.A.detach().clone());self.B=torch.nn.Parameter(p.B.detach().clone());self.scale=p.scale
        def forward(self,x):
            y=self.base(x);return (y.float()+F.linear(F.linear(x.float(),self.A),self.B)*self.scale).to(y.dtype)
    for b in adapter.blocks:
        if b.kind=='linear_attention':
            c=model.model.layers[b.index].linear_attn;c.causal_conv1d_fn=None;c.chunk_gated_delta_rule=h.torch_chunk_gated_delta_rule
        for k,p in b.adapters.items():
            parent,leaf=k.rsplit('.',1);owner=model.model.layers[b.index].get_submodule(parent);ref=Projection(getattr(owner,leaf),p);setattr(owner,leaf,ref);refs[f'{b.index}:{k}']=ref
    return model,refs


def loss(model,ids,labels):
    x=model.model.embed_tokens(ids).detach().requires_grad_(True);logits=model(inputs_embeds=x,use_cache=False,output_router_logits=False).logits
    return F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum'),x


@pytest.mark.parametrize('quant',['none','int4-sym-g128'])
@pytest.mark.parametrize('nonzero_b',[False,True])
def test_native_whole_text_input_and_lora_vjp(tmp_path,quant,nonzero_b):
    torch.set_num_threads(1);source=make_tiny(tmp_path/'source');a=Qwen35MoeAdapter(source,quantization=quant)
    if nonzero_b:
        with torch.no_grad():
            for p in flatten(a).values():p.B.normal_(0,.03)
    model,refs=native(a);ids=torch.tensor([[1,2,3,4,5,6]]);labels=torch.tensor([[-100,-100,3,4,5,6]]);expected,x=loss(model,ids,labels);(expected/4).backward()
    session=SlotSession(a.blocks,'cpu',host_budget_bytes=1024**3,device_budget_bytes=1024**3)
    with session.run():actual,grad=microbatch(a,session,ids,labels,normalization=4,token_chunk=2)
    assert actual==pytest.approx(float(expected.detach()/4),abs=3e-6,rel=3e-6)
    torch.testing.assert_close(grad,x.grad,atol=.002,rtol=.04);assert torch.count_nonzero(grad[:,:2])>0
    for k,p in flatten(a).items():
        for f in ['A','B']:torch.testing.assert_close(getattr(p,'grad_'+f),getattr(refs[k],f).grad,atol=.002,rtol=.04)
    assert all(t.device.type=='meta' for b in a.blocks for t in b.kwargs['native_template'].parameters());session.close();a.close()


@pytest.mark.parametrize('quant',['none','int4-sym-g128'])
def test_native_two_updates_accumulation_moments(tmp_path,quant):
    torch.set_num_threads(1);a=Qwen35MoeAdapter(make_tiny(tmp_path/'source'),quantization=quant);model,refs=native(a);projs=flatten(a)
    opt=LoRAOptimizer(lr=.001,weight_decay=.1);oracle=torch.optim.AdamW([x for p in refs.values() for x in [p.A,p.B]],lr=.001,weight_decay=.1,foreach=False)
    session=SlotSession(a.blocks,'cpu',host_budget_bytes=1024**3,device_budget_bytes=1024**3);rows=[(torch.tensor([[1,2,3,4,5,6]]),torch.tensor([[-100,-100,3,4,5,6]])),(torch.tensor([[2,4,6,8]]),torch.tensor([[-100,-100,6,8]]))]
    for _ in range(2):
        opt.zero_grad(projs);oracle.zero_grad(set_to_none=True);expected=sum(loss(model,*row)[0] for row in rows)/6;expected.backward()
        with session.run():actual=sum(microbatch(a,session,*row,normalization=6,token_chunk=2,accumulate=i>0)[0] for i,row in enumerate(rows))
        assert actual==pytest.approx(float(expected.detach()),abs=3e-6,rel=3e-6)
        for k,p in projs.items():
            for f in ['A','B']:
                torch.testing.assert_close(getattr(p,'grad_'+f),getattr(refs[k],f).grad,atol=.002,rtol=.04);getattr(refs[k],f).grad.copy_(getattr(p,'grad_'+f))
        opt.step(projs);oracle.step()
        for k,p in projs.items():
            for f in ['A','B']:
                v=getattr(refs[k],f);torch.testing.assert_close(getattr(p,f),v,atol=2e-7,rtol=2e-6)
                torch.testing.assert_close(getattr(p,'m_'+f),oracle.state[v]['exp_avg'],atol=2e-7,rtol=2e-6);torch.testing.assert_close(getattr(p,'v_'+f),oracle.state[v]['exp_avg_sq'],atol=2e-7,rtol=2e-6)
    session.close();a.close()


@pytest.mark.parametrize('change',[{'model_type':'qwen3_5'},{'num_experts_per_tok':4},{'output_router_logits':True},{'tie_word_embeddings':True},{'attention_dropout':.1},{'attn_output_gate':False},{'num_hidden_layers':True}])
def test_explicit_config_rejections(tmp_path,change):
    source=make_tiny(tmp_path/'source');p=source/'config.json';cfg=json.loads(p.read_text());cfg.update(change);p.write_text(json.dumps(cfg))
    with pytest.raises((ValueError,TypeError)):inspect_qwen35_moe(source)


@pytest.mark.parametrize('kind',['missing','extra','shape','dtype'])
def test_exact_source_inventory(tmp_path,kind):
    source=make_tiny(tmp_path/'source');p=source/'model.safetensors';d=load_file(str(p));key='model.language_model.layers.0.mlp.experts.gate_up_proj'
    if kind=='missing':d.pop(key)
    if kind=='extra':d['unknown.weight']=torch.ones(1,dtype=torch.bfloat16)
    if kind=='shape':d[key]=d[key][:2]
    if kind=='dtype':d[key]=d[key].float()
    save_file(d,str(p))
    with pytest.raises(ValueError):inspect_qwen35_moe(source)


@pytest.mark.parametrize('kw',[{'rank':True},{'alpha':float('nan')},{'quantization':'int8-sym-g128'},{'host_budget_bytes':1},{'device_budget_bytes':1}])
def test_early_constructor_guards(tmp_path,kw):
    source=make_tiny(tmp_path/'source')
    with pytest.raises((ValueError,MemoryError)):Qwen35MoeAdapter(source,**kw)


def test_cli_fixture_and_inspection(tmp_path,capsys,monkeypatch):
    from ram_vram_finetune.cli import main
    source=tmp_path/'cli-source'
    import sys
    monkeypatch.setattr(sys,'argv',['rvft','make-tiny','--output',str(source),'--adapter','qwen35_moe','--layers','4']);assert main()==0
    monkeypatch.setattr(sys,'argv',['rvft','inspect','--model',str(source),'--adapter','qwen35_moe']);assert main()==0
    assert 'qwen35_moe' in capsys.readouterr().out


@pytest.mark.parametrize('branch',['vision','mtp'])
def test_declared_excluded_branch_cannot_disappear(tmp_path,branch):
    source=make_tiny(tmp_path/'source');p=source/'config.json';d=json.loads(p.read_text())
    if branch=='vision':d['vision_config']={}
    else:d['mtp_num_hidden_layers']=1
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='declared'):inspect_qwen35_moe(source)


def add_mtp(source):
    p=source/'config.json';cfg=json.loads(p.read_text());cfg['mtp_num_hidden_layers']=1;cfg['mtp_use_dedicated_embeddings']=False;p.write_text(json.dumps(cfg))
    p=source/'model.safetensors';d=load_file(str(p));prefix='model.language_model.layers.3.'
    d.update({'mtp.layers.0.'+k.removeprefix(prefix):v.clone() for k,v in list(d.items()) if k.startswith(prefix)})
    d.update({'mtp.fc.weight':torch.ones(16,32,dtype=torch.bfloat16),**{'mtp.'+k+'.weight':torch.zeros(16,dtype=torch.bfloat16) for k in ['norm','pre_fc_norm_embedding','pre_fc_norm_hidden']}});save_file(d,str(p))


def test_excluded_mtp_finite_and_identity_bound(tmp_path):
    source=make_tiny(tmp_path/'source');add_mtp(source);a=Qwen35MoeAdapter(source,quantization='int4-sym-g128');first=a.identity;a.close();p=source/'model.safetensors';d=load_file(str(p));d['mtp.fc.weight'][0,0]+=1;save_file(d,str(p));a=Qwen35MoeAdapter(source,quantization='int4-sym-g128');assert a.identity!=first;a.close();d['mtp.fc.weight'][0,0]=float('nan');save_file(d,str(p))
    with pytest.raises(ValueError,match='nonfinite excluded'):Qwen35MoeAdapter(source,quantization='int4-sym-g128')


def test_source_mutation_during_load_rejected(tmp_path,monkeypatch):
    from ram_vram_finetune.quantization.storage import WeightLoader
    source=make_tiny(tmp_path/'source');original=WeightLoader.read;done=[]
    def changed(self,key):
        value=original(self,key)
        if not done:
            p=source/'config.json';d=json.loads(p.read_text());d['source_mutated']=True;p.write_text(json.dumps(d));done.append(True)
        return value
    monkeypatch.setattr(WeightLoader,'read',changed)
    with pytest.raises(ValueError,match='source changed during'):Qwen35MoeAdapter(source,quantization='int4-sym-g128')



def test_conv_retains_raw_bf16_and_none_is_not_quantized(tmp_path):
    source=make_tiny(tmp_path/'source');q=Qwen35MoeAdapter(source,quantization='int4-sym-g128');raw=load_file(str(source/'model.safetensors'))
    for block in q.blocks:
        if block.kind=='linear_attention':
            key='linear_attn.conv1d.weight';item=block.storage_specs[key];assert item['spec']['codec']=='raw';assert item['spec']['dtype']=='bfloat16';assert torch.equal(block.host_weights[item['parts']['value']],raw[f'model.language_model.layers.{block.index}.'+key])
    q.close();a=Qwen35MoeAdapter(source,quantization='none');assert a.storage_metadata is None;a.close()
    cfg=TrainConfig(str(source),str(source/'tokens.jsonl'),str(tmp_path/'job'),adapter='qwen35_moe',steps=1,quantization='none')
    run(cfg);meta=json.loads((tmp_path/'job/adapter/metadata.json').read_text());assert meta.get('quantized_base') is None;assert not (tmp_path/'job/peft-adapter/rvft_quantized_base.json').exists()


@pytest.mark.parametrize('location',['root','text'])
def test_external_quantization_config_rejected(tmp_path,location):
    source=make_tiny(tmp_path/'source');p=source/'config.json';cfg=json.loads(p.read_text())
    if location=='root':cfg['quantization_config']={}
    else:cfg={'model_type':'qwen3_5_moe','text_config':{**cfg,'quantization_config':{'quant_method':'external'}}}
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='external compressed'):inspect_qwen35_moe(source)


def test_source_index_must_match_physical_inventory(tmp_path):
    source=make_tiny(tmp_path/'source');d=load_file(str(source/'model.safetensors'));(source/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'wrong.safetensors' for k in d}}))
    with pytest.raises(ValueError,match='exact source shard index'):inspect_qwen35_moe(source)
