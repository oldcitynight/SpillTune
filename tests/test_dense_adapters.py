"""Whole native HF models construct their own mask, positions, rotary and loss."""
import copy
import dataclasses
import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ram_vram_finetune.adapters.dense import hf_classes
from ram_vram_finetune.adapters.registry import resolve
from ram_vram_finetune.config import TrainConfig
from ram_vram_finetune.finetune.optimizer import LoRAOptimizer
from ram_vram_finetune.fixture import make_tiny
from ram_vram_finetune.memory import SlotSession
from ram_vram_finetune.training.engine import flatten, microbatch, forward, run


def setup(tmp_path, family, device='cpu', dtype=torch.float32, config_patch=None):
    torch.set_num_threads(1)
    path = make_tiny(tmp_path / family, adapter=family, layers=2)
    if config_patch:
        raw=json.loads((path/'config.json').read_text());raw.update(config_patch)
        if raw.pop('_omit_head_dim',False):raw.pop('head_dim',None)
        (path/'config.json').write_text(json.dumps(raw))
        if family=='qwen3' and raw.get('tie_word_embeddings') is False:
            from safetensors.torch import save_file
            owned_weights=load_file(str(path/'model.safetensors'))
            owned_weights['lm_head.weight']=torch.randn_like(owned_weights['model.embed_tokens.weight'])*.1
            save_file(owned_weights,str(path/'model.safetensors'))
    if dtype != torch.float32:
        from safetensors.torch import save_file
        weights = load_file(str(path / 'model.safetensors'))
        save_file({k: v.to(dtype) for k, v in weights.items()}, str(path / 'model.safetensors'))
    adapter = resolve(family)(path, device=device)
    Config, _, Model = hf_classes(family)
    native_config=Config(**json.loads((path/'config.json').read_text()));native_config._attn_implementation='eager'
    native = Model(native_config).to(device=device).eval().requires_grad_(False)
    # Assign checkpoint parameter dtype without downcasting native FP32 RoPE buffers.
    native_weights={k:v.to(device) for k,v in load_file(str(path/'model.safetensors')).items()}
    if native_config.tie_word_embeddings:native_weights['lm_head.weight']=native_weights['model.embed_tokens.weight']
    native.load_state_dict(native_weights,strict=True,assign=True)
    if native_config.tie_word_embeddings:
        native.tie_weights();assert native.lm_head.weight is native.model.embed_tokens.weight
    assert native.model.rotary_emb.inv_freq.dtype==torch.float32
    reference = {}

    class Projection(torch.nn.Module):
        def __init__(self, original, p):
            super().__init__()
            self.original = original
            self.A = torch.nn.Parameter(p.A.detach().clone())
            self.B = torch.nn.Parameter(p.B.detach().clone())
            self.scale = p.scale

        def forward(self, x):
            base = self.original(x)
            update = F.linear(F.linear(x.float(), self.A), self.B)
            return (base.float() + update * self.scale).to(base.dtype)

    for block in adapter.blocks:
        for target, p in block.adapters.items():
            parent_path, leaf = target.rsplit('.', 1)
            parent = native.model.layers[block.index].get_submodule(parent_path)
            wrapped = Projection(getattr(parent, leaf), p)
            setattr(parent, leaf, wrapped)
            reference[f'{block.index}:{target}'] = wrapped
    slot = SlotSession(adapter.blocks, device, host_budget_bytes=1024**3, device_budget_bytes=1024**3)
    return path, adapter, native, reference, slot


def native_loss(native, ids, labels):
    embedded = native.model.embed_tokens(ids.to(native.device)).detach().requires_grad_(True)
    # No mask or rotary values from adapter; the native top-level builds both.
    logits = native(inputs_embeds=embedded, use_cache=False).logits
    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].to(logits.device).reshape(-1), reduction='sum', ignore_index=-100)
    return loss, embedded


def close(a, b, dtype=torch.float32):
    torch.testing.assert_close(a, b, rtol=2e-5 if dtype == torch.float32 else 2e-2,
                               atol=2e-6 if dtype == torch.float32 else 2e-3)


@pytest.mark.parametrize('family', ['llama', 'qwen2','qwen3'])
@pytest.mark.parametrize('nonzero_b', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_whole_hf_loss_every_gradient(tmp_path, family, nonzero_b, dtype, record_property):
    _, adapter, native, refs, slot = setup(tmp_path, family, dtype=dtype)
    projections = flatten(adapter)
    if nonzero_b:
        with torch.no_grad():
            for key, p in projections.items():
                p.B.normal_(0, .03); refs[key].B.copy_(p.B)
    ids, labels = torch.tensor([[1, 3, 5, 7, 9]]), torch.tensor([[-100, -100, 5, 7, 9]])
    loss, embedded = native_loss(native, ids, labels)
    (loss / 3).backward()
    with slot.run():
        actual, gradient = microbatch(adapter, slot, ids, labels, normalization=3, token_chunk=2)
    assert actual == pytest.approx(float(loss.detach()/3), rel=2e-6, abs=2e-6)
    close(gradient, embedded.grad, dtype)
    errors = [float((gradient-embedded.grad).abs().max())]
    for key, p in projections.items():
        close(p.grad_A, refs[key].A.grad, dtype); close(p.grad_B, refs[key].B.grad, dtype)
        errors += [float((p.grad_A-refs[key].A.grad).abs().max()), float((p.grad_B-refs[key].B.grad).abs().max())]
    record_property('max_abs_gradient_error', max(errors))
    record_property('loss_abs_error', abs(actual - float(loss.detach()/3)))
    slot.close(); adapter.close()


@pytest.mark.parametrize('family', ['llama', 'qwen2','qwen3'])
def test_two_adamw_updates_accumulation_and_moments(tmp_path, family, record_property):
    _, adapter, native, refs, slot = setup(tmp_path, family)
    projections = flatten(adapter)
    native_optimizer = torch.optim.AdamW([t for r in refs.values() for t in (r.A, r.B)], lr=.001, weight_decay=.1, foreach=False)
    optimizer = LoRAOptimizer(lr=.001, weight_decay=.1)
    frozen = {f'{b.index}:{k}': v.clone() for b in adapter.blocks for k, v in b.host_weights.items()}
    rows = [(torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[-100, -100, 3, 4, 5]])),
            (torch.tensor([[3, 6, 9]]), torch.tensor([[-100, -100, 9]]))]
    worst = 0.
    for _ in range(2):
        native_optimizer.zero_grad(set_to_none=True)
        loss = sum(native_loss(native, *row)[0] for row in rows)/4
        loss.backward()
        optimizer.zero_grad(projections)
        with slot.run():
            actual = sum(microbatch(adapter, slot, *row, normalization=4, token_chunk=1, accumulate=i>0)[0]
                         for i, row in enumerate(rows))
            assert actual == pytest.approx(float(loss.detach()), rel=3e-6)
            for key, p in projections.items():
                close(p.grad_A, refs[key].A.grad); close(p.grad_B, refs[key].B.grad)
            optimizer.step(projections)
        native_optimizer.step()
        for key, p in projections.items():
            r = refs[key]
            for name in ('A', 'B'):
                close(getattr(p, name), getattr(r, name))
                worst = max(worst, float((getattr(p, name)-getattr(r, name)).abs().max().detach()))
                for prefix, field in [('m_', 'exp_avg'), ('v_', 'exp_avg_sq')]:
                    close(getattr(p, prefix+name), native_optimizer.state[getattr(r, name)][field])
    assert all(torch.equal(v, frozen[f'{b.index}:{k}']) for b in adapter.blocks for k, v in b.host_weights.items())
    record_property('max_abs_update_error', worst)
    slot.close(); adapter.close()


@pytest.mark.parametrize('family', ['llama', 'qwen2','qwen3'])
@pytest.mark.parametrize('valid', [[[1,1,1,1,1]], [[1,1,1,0,0]]])
def test_independent_native_mask_rotary_whole_model(tmp_path, family, valid):
    _, adapter, native, _, slot = setup(tmp_path, family)
    ids = torch.tensor([[1,2,3,4,5]])
    mask = torch.tensor(valid, dtype=torch.bool)
    embedded = F.embedding(ids, adapter.embedding).detach().requires_grad_(True)
    positions, masks = adapter.positions_and_masks(mask)
    expected = native.model(inputs_embeds=embedded, attention_mask=mask, use_cache=False).last_hidden_state
    with slot.run():
        actual = adapter.terminal_norm(forward(adapter, slot, embedded, positions, masks))
        close(actual, expected)
        torch.autograd.grad(actual.sum(), embedded)
    slot.close(); adapter.close()


@pytest.mark.parametrize('family', ['llama', 'qwen2','qwen3'])
def test_exact_resume_all_state_and_artifacts(tmp_path, family):
    path = make_tiny(tmp_path / 'model', adapter=family, layers=2)
    config = TrainConfig(str(path), str(path/'tokens.jsonl'), str(tmp_path/'full'), adapter=family, accumulation=2, steps=3)
    full = run(config)
    cancelled = run(dataclasses.replace(config, output=str(tmp_path/'cancel')), cancel_after_step=1)
    resumed = run(dataclasses.replace(config, output=str(tmp_path/'resume')), resume=tmp_path/'cancel/checkpoint-1')
    assert cancelled['status'] == 'cancelled' and full['status'] == resumed['status'] == 'completed'
    left = load_file(str(tmp_path/'full/checkpoint-3/state.safetensors'))
    right = load_file(str(tmp_path/'resume/checkpoint-3/state.safetensors'))
    assert left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)
    assert full['metrics'][-1]['loss'] == resumed['metrics'][-1]['loss']
    artifact = load_file(str(tmp_path/'resume/peft-adapter/adapter_model.safetensors'))
    assert len(artifact) == 12 and all(k.startswith('base_model.model.model.layers.') for k in artifact)


@pytest.mark.parametrize('family', ['llama', 'qwen2'])
@pytest.mark.parametrize('change', [{'tie_word_embeddings': True}, {'attention_dropout': .1},
    {'rope_parameters': {'rope_type': 'linear', 'rope_theta': 10000., 'factor': 2.}},
    {'sliding_window': 16,'use_sliding_window':True}, {'hidden_act': 'relu'}, {'attention_bias': True}, {'mlp_bias': True}])
def test_reject_unvalidated_capabilities_before_payload(tmp_path, family, change):
    path = make_tiny(tmp_path/'model', adapter=family, layers=1)
    file = path/'config.json'; raw = json.loads(file.read_text()); raw.update(change); file.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='unsupported_capability'):
        resolve(family)(path)


@pytest.mark.parametrize('family', ['llama', 'qwen2','qwen3'])
def test_reject_left_padding_position_overflow_and_wrong_shapes(tmp_path, family):
    path, adapter, _, _, slot = setup(tmp_path, family)
    for mask in (torch.tensor([[False, True]]), torch.tensor([[True, False, True]]), torch.ones((1,65), dtype=torch.bool)):
        with pytest.raises(ValueError, match='unsupported_capability'):
            adapter.positions_and_masks(mask)
    slot.close(); adapter.close()
    raw = json.loads((path/'config.json').read_text()); raw['hidden_size'] = 24; (path/'config.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='shape mismatch'):
        resolve(family)(path)


@pytest.mark.parametrize('family', ['llama', 'qwen2','qwen3'])
def test_local_chat_tokenizer_reload_and_training(tmp_path, family):
    from ram_vram_finetune.training.engine import load_rows
    path = make_tiny(tmp_path/'model', adapter=family, layers=2)
    config = TrainConfig(str(path), str(path/'chat.jsonl'), str(tmp_path/'chat'), adapter=family, steps=1)
    rows = load_rows(config)
    ids, labels = rows[0]
    assert (labels != -100).sum() >= 3
    assert int(ids.max()) < 32
    assert torch.equal(ids[labels!=-100], labels[labels!=-100])
    assert run(config)['status'] == 'completed'


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('window',[1,2])
@pytest.mark.parametrize('legacy_head',[False,True])
def test_explicit_disabled_qwen2_window_independent_full_hf_two_updates(tmp_path,dtype,window,legacy_head):
    path,a,native,refs,slot=setup(tmp_path,'qwen2',dtype=dtype,
        config_patch={'use_sliding_window':False,'sliding_window':window,'max_window_layers':0,'_omit_head_dim':legacy_head})
    assert a.config.head_dim==8
    if legacy_head:assert 'head_dim' not in json.loads((path/'config.json').read_text())
    assert a.config.sliding_window is None and native.config.sliding_window is None
    assert native.config.layer_types==['full_attention']*2
    assert json.loads((path/'config.json').read_text())['sliding_window']==window
    # Prove the input extends past the window and the active HF alternative
    # changes outputs. No adapter mask/rotary is passed to either HF model.
    Config,_,Model=hf_classes('qwen2');raw=json.loads((path/'config.json').read_text())
    raw.update(use_sliding_window=True,layer_types=['sliding_attention']*2)
    cfg=Config(**raw);cfg._attn_implementation='eager'
    enabled=Model(cfg).eval();enabled.load_state_dict(load_file(str(path/'model.safetensors')),assign=True)
    ids=torch.tensor([[1,2,3,4,5]])
    assert not torch.equal(native(input_ids=ids,use_cache=False).logits,enabled(input_ids=ids,use_cache=False).logits)
    with torch.no_grad():
        for key,p in flatten(a).items():p.B.normal_(0,.03);refs[key].B.copy_(p.B)
    opt=LoRAOptimizer(lr=.001,weight_decay=.1);expected=torch.optim.AdamW([t for r in refs.values() for t in (r.A,r.B)],lr=.001,weight_decay=.1,foreach=False)
    rows=[(ids,torch.tensor([[-100,-100,3,4,5]])),(torch.tensor([[3,6,9]]),torch.tensor([[-100,-100,9]]))]
    def prefix_loss(ids,labels):
        x=native.model.embed_tokens(ids).detach().requires_grad_(True)
        # Select only logits used by shifted CE through the native API. This
        # matches the head GEMM row count; native still owns mask and rotary.
        logits=native(inputs_embeds=x,use_cache=False,logits_to_keep=torch.arange(ids.shape[1]-1)).logits
        return F.cross_entropy(logits.float().reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum'),x
    for _ in range(2):
        expected.zero_grad(set_to_none=True);losses=[prefix_loss(*r) for r in rows]
        loss=sum(x[0] for x in losses)/4;loss.backward();opt.zero_grad(flatten(a))
        with slot.run():
            actual=[microbatch(a,slot,*r,normalization=4,token_chunk=8,accumulate=i>0) for i,r in enumerate(rows)]
            assert abs(sum(x[0] for x in actual)-float(loss.detach()))<2e-6
            pairs=[(x[1],y[1].grad) for x,y in zip(actual,losses)]
            pairs += [(getattr(p,'grad_'+f),getattr(refs[k],f).grad) for k,p in flatten(a).items() for f in ['A','B']]
            for x,y in pairs:
                close(x,y,dtype)
                assert float((x-y).float().norm()/y.float().norm().clamp_min(1e-12))<(.05 if dtype==torch.bfloat16 else .0001)
            if dtype==torch.bfloat16:
                for k,p in flatten(a).items():refs[k].A.grad=p.grad_A.detach().clone();refs[k].B.grad=p.grad_B.detach().clone()
            opt.step(flatten(a))
        expected.step()
        for k,p in flatten(a).items():
            for f in ['A','B']:
                ref=getattr(refs[k],f)
                for actual,want in [(getattr(p,f),ref),(getattr(p,'m_'+f),expected.state[ref]['exp_avg']),(getattr(p,'v_'+f),expected.state[ref]['exp_avg_sq'])]:
                    torch.testing.assert_close(actual,want,rtol=0 if dtype==torch.bfloat16 else 2e-5,atol=0 if dtype==torch.bfloat16 else 2e-6)
    slot.close();a.close()


@pytest.mark.parametrize('patch',[{'use_sliding_window':True,'sliding_window':2},
    {'use_sliding_window':False,'sliding_window':2,'layer_types':['sliding_attention','full_attention']},
    {'use_sliding_window':0,'sliding_window':2},{'use_sliding_window':False,'sliding_window':-1}])
def test_qwen2_active_unresolved_or_malformed_window_still_rejected(tmp_path,patch):
    path=make_tiny(tmp_path/'model',adapter='qwen2',layers=2)
    raw=json.loads((path/'config.json').read_text());raw.update(patch);(path/'config.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='unsupported_capability'):resolve('qwen2')(path)
