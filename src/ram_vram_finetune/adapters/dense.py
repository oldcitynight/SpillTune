"""Strict dense Llama/Qwen2/Qwen3 adapters: GQA, standard RMSNorm and full RoPE.

Independent functional tensor graph, validated against native Transformers
whole-model forward/backward. No resident decoder modules, caches or downloads.
"""
from ..quantization.codecs import FORMATS

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ..finetune.lora import create_lora_for_block
from .targets import ATTENTION_TARGETS
from .protocol import BlockPlan
from ..quantization.storage import scan_entries,WeightLoader,estimated_bytes
from ..quantization.codecs import materialize,flatten_records


def hf_classes(family):
    if family == 'llama':
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
        return LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM
    if family == 'qwen2':
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
        return Qwen2Config, Qwen2DecoderLayer, Qwen2ForCausalLM
    if family == 'qwen3':
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM
        return Qwen3Config, Qwen3DecoderLayer, Qwen3ForCausalLM
    raise ValueError('unsupported_architecture: expected llama, qwen2 or qwen3')


def inspect_dense(path, family):
    root = Path(path)
    raw = json.loads((root / 'config.json').read_text(encoding="utf-8"))
    if raw.get('model_type') != family:
        raise ValueError(f'unsupported_architecture: expected {family}')
    if family=='qwen2' and 'use_sliding_window' in raw and type(raw['use_sliding_window']) is not bool:
        raise ValueError('unsupported_capability: use_sliding_window must be boolean')
    if family=='qwen3' and (type(raw.get('tie_word_embeddings',False)) is not bool):
        raise ValueError('unsupported_capability: tie_word_embeddings must be explicit boolean')
    Config, Layer, _ = hf_classes(family)
    config = Config(**raw)
    config._attn_implementation = 'eager'
    # Qwen2 legacy HF configs omit head_dim; native attention uses H/heads.
    # Explicit malformed values are still rejected, not silently replaced.
    if family=='qwen2' and not hasattr(config,'head_dim'):
        h,n=config.hidden_size,config.num_attention_heads
        if type(h)is not int or type(n)is not int or min(h,n)<1 or h%n:
            raise ValueError('unsupported_capability: cannot resolve legacy Qwen2 head dimension')
        config.head_dim=h//n
    tied=(family=='qwen3' and config.tie_word_embeddings is True)
    if config.tie_word_embeddings and not tied:
        raise ValueError('unsupported_capability: tied embeddings')
    if config.attention_dropout != 0 or config.hidden_act != 'silu':
        raise ValueError('unsupported_capability: dropout/activation')
    if getattr(config, 'attention_bias', False) or getattr(config, 'mlp_bias', False):
        raise ValueError('unsupported_capability: configurable projection biases (Qwen2 fixed QKV bias is supported)')
    if getattr(config, 'pretraining_tp', 1) != 1:
        raise ValueError('unsupported_capability: pretraining_tp')
    rope = config.rope_parameters
    if set(rope) - {'rope_type', 'rope_theta'} or rope.get('rope_type') != 'default':
        raise ValueError('unsupported_capability: only default full rotary embeddings')
    if not math.isfinite(rope['rope_theta']) or rope['rope_theta'] <= 0:
        raise ValueError('invalid rotary base')
    raw_window=raw.get('sliding_window')
    disabled_qwen2=(family=='qwen2' and raw.get('use_sliding_window') is False
        and config.use_sliding_window is False and config.sliding_window is None)
    if raw_window is not None and (type(raw_window) is not int or raw_window<1):
        raise ValueError('unsupported_capability: invalid sliding window metadata')
    if (raw_window is not None and not disabled_qwen2) or getattr(config,'use_sliding_window',False) or getattr(config,'sliding_window',None) is not None:
        raise ValueError('unsupported_capability: active or unresolved sliding window')
    if not math.isfinite(config.rms_norm_eps) or config.rms_norm_eps <= 0:
        raise ValueError('unsupported_capability: invalid norm epsilon')
    types = getattr(config, 'layer_types', ['full_attention'] * config.num_hidden_layers)
    if len(types) != config.num_hidden_layers or set(types) != {'full_attention'}:
        raise ValueError('unsupported_capability: layer types')
    dims = [config.hidden_size, config.vocab_size, config.intermediate_size, config.num_hidden_layers,
            config.num_attention_heads, config.num_key_value_heads, config.head_dim, config.max_position_embeddings]
    if any(type(x) is not int or x < 1 for x in dims) or config.head_dim % 2 or config.num_attention_heads % config.num_key_value_heads:
        raise ValueError('unsupported_capability: invalid GQA/head/position dimensions')
    with torch.device('meta'):
        expected = Layer(config, 0).state_dict()
    shapes = {'model.embed_tokens.weight': [config.vocab_size, config.hidden_size],
              'model.norm.weight': [config.hidden_size], 'lm_head.weight': [config.vocab_size, config.hidden_size]}
    for i in range(config.num_hidden_layers):
        shapes.update({f'model.layers.{i}.{k}': list(v.shape) for k, v in expected.items()})
    if tied:shapes.pop('lm_head.weight')
    entries = scan_entries(root,family)
    if set(entries) != set(shapes):
        raise ValueError(f'unsupported_capability: checkpoint key inventory; missing={sorted(shapes.keys()-entries.keys())}; extra={sorted(entries.keys()-shapes.keys())}')
    if any(entries[k]['shape'] != shape for k, shape in shapes.items()):
        raise ValueError('checkpoint/config shape mismatch')
    dtypes = {v['dtype'] for v in entries.values()}
    if len(dtypes) != 1 or not dtypes <= {'F32', 'BF16'}:
        raise ValueError('unsupported_capability: homogeneous FP32 or BF16 required')
    itemsize = 4 if dtypes == {'F32'} else 2
    size = lambda k: entries[k].get('stored_bytes',math.prod(shapes[k]) * itemsize)
    return {'quantized_export': {'version':1,'formats':[] if family=='qwen3' else list(FORMATS),'rejection_reason':'Qwen3 compressed path not validated' if family=='qwen3' else ''}, 'architecture': family, 'layers': config.num_hidden_layers, 'text_config': raw,
            'entries': entries, 'required': sorted(entries), 'config': config,
            'quantization_matrix_keys':[k for k,v in entries.items() if k.startswith('model.layers.') and len(v['shape'])==2 and math.prod(v['shape'])>=128],
            'quantization_fp32_keys':[],
            'host_weight_bytes': sum(size(k) for k in shapes if k not in {'model.norm.weight', 'lm_head.weight'}),
            'head_bytes': size('model.embed_tokens.weight' if tied else 'lm_head.weight'),
            'source_aliases':{'lm_head.weight':'model.embed_tokens.weight'} if tied else {}}


def rms(hidden, weight, eps):
    value = hidden.float()
    return weight * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(hidden.dtype)


def dense_block(hidden_states, position_embeddings, attention_mask, *, heads, kv_heads, head_dim,
                eps, lora_q, lora_v, lora_o, qk_norm=False, feed_forward=None, **weights):
    def linear(x, path, lora=None):
        weight, bias = weights[path + '.weight'], weights.get(path + '.bias')
        return lora.forward(x, weight, bias) if lora is not None else F.linear(x, weight, bias)
    residual = hidden_states
    x = rms(hidden_states, weights['input_layernorm.weight'], eps)
    shape = (*x.shape[:-1], -1, head_dim)
    q = linear(x, 'self_attn.q_proj', lora_q).view(shape)
    k = linear(x, 'self_attn.k_proj').view(shape)
    v = linear(x, 'self_attn.v_proj', lora_v).view(shape).transpose(1, 2)
    if qk_norm:
        # Match native Qwen3 layout as well as real-valued math: BF16 VJP
        # reduction/cast order can depend on strides after transpose.
        q = rms(q, weights['self_attn.q_norm.weight'], eps)
        k = rms(k, weights['self_attn.k_norm.weight'], eps)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    cos, sin = (value.unsqueeze(1) for value in position_embeddings)
    def rotate(x):
        left, right = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-right, left), dim=-1) * sin
    q, k = rotate(q), rotate(k)
    def repeat(x):
        b, h, s, d = x.shape
        return x[:, :, None].expand(b, h, heads // kv_heads, s, d).reshape(b, heads, s, d)
    k, v = repeat(k), repeat(v)
    scores = (q @ k.transpose(-2, -1)) * head_dim ** -0.5
    probabilities = torch.softmax(scores + attention_mask, dim=-1, dtype=torch.float32).to(q.dtype)
    attended = (probabilities @ v).transpose(1, 2).contiguous().reshape(*x.shape[:-1], heads * head_dim)
    x = residual + linear(attended, 'self_attn.o_proj', lora_o)
    residual = x
    x = rms(x, weights['post_attention_layernorm.weight'], eps)
    if feed_forward is not None:return residual + feed_forward(x,weights)
    return residual + linear(F.silu(linear(x, 'mlp.gate_proj')) * linear(x, 'mlp.up_proj'), 'mlp.down_proj')


def _new_loader(path, report, mode, device, host_budget_bytes):
    """Metadata-only budget gate; direct Q4 avoids pageable packed duplicates.

    Larger shared arenas reduce power-of-two host allocator waste on dense
    FFN matrices. The CUDA session later reuses these exact owners. The source
    embedding keeps its read-only mapping; no model payload is copied on disk.
    """
    direct = mode == 'int4-sym-g128' and torch.device(device).type == 'cuda'
    loader = WeightLoader(path, report, mode,
        pin_keys=[k for k in report['required'] if k.startswith('model.layers.')] if direct else (),
        host_budget_bytes=host_budget_bytes,
        pinned_arena_limit=(512 if direct else 128)*1024**2)
    host_bytes = sum(estimated_bytes(report, mode, k) for k in report['required']
                     if k not in {'model.norm.weight', 'lm_head.weight'})
    if direct:
        host_bytes = max(host_bytes, sum(loader.pinned_capacities)
                         + estimated_bytes(report, mode, 'model.embed_tokens.weight'))
    if host_bytes > host_budget_bytes:
        raise MemoryError('resource_budget: host weights and actual pinned capacity exceed allowance before loading')
    return loader, direct


class DenseDecoderAdapter:
    family = None

    def __init__(self, path, *, rank=2, alpha=4., device='cpu', host_budget_bytes=1024**3, device_budget_bytes=2*1024**3,
                 quantization='none',quantized_compute='bf16'):
        report = inspect_dense(path, self.family)
        if self.family=='qwen3' and quantization!='none':
            raise ValueError('unsupported_quantization: Qwen3 BF16/FP32 source only; no validated compressed path')
        if quantized_compute!='bf16':raise ValueError('unsupported_quantization: only storage plus BF16 compute')
        loader,direct_pinned=_new_loader(path,report,quantization,device,host_budget_bytes)
        if estimated_bytes(report,quantization,report['source_aliases'].get('lm_head.weight','lm_head.weight')) > device_budget_bytes:
            raise MemoryError('resource_budget: model exceeds host/head allowance before loading')
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('dependency: explicit CUDA requested but unavailable')
        self.config = report['config']
        state = {}
        for key in report['required']:
            record=loader.read(key)
            state[key] = materialize(record,self.device) if key in {'lm_head.weight', 'model.norm.weight'} else record
        self.identity = loader.finish()
        self.storage_metadata={'format':quantization,'compute':'bf16' if quantization!='none' else 'source',
            'storage_identity':loader.storage_identity,'source_identity':loader.source_identity,
            'total_stored_bytes':loader.stored_bytes,'max_source_tensor_bytes':loader.max_source_tensor_bytes,
            'full_bf16_decoder_retained':False,'quantized_scope':'decoder matrices >=128 elements; boundaries/norm/bias excluded'} if quantization!='none' else None
        self.source_aliases=report['source_aliases']
        self.embedding=materialize(state.pop('model.embed_tokens.weight'))
        self.norm=state.pop('model.norm.weight')
        if self.source_aliases:
            self.head=self.embedding.to(self.device)
        else:self.head=state.pop('lm_head.weight')
        assert not self.embedding.requires_grad and not self.head.requires_grad
        self.blocks = []
        for index in range(self.config.num_hidden_layers):
            prefix = f'model.layers.{index}.'
            weights = {k.removeprefix(prefix): state.pop(k) for k in list(state) if k.startswith(prefix)}
            adapters = create_lora_for_block(index, 'full_attention', {k:v.shape for k,v in weights.items()}, rank=rank, alpha=alpha,
                                             device=self.device, targets=ATTENTION_TARGETS['full_attention'], architecture=self.family)
            kwargs = {'heads': self.config.num_attention_heads, 'kv_heads': self.config.num_key_value_heads,
                      'head_dim': self.config.head_dim, 'eps': self.config.rms_norm_eps,'qk_norm':self.family=='qwen3',
                      **{f'lora_{p}': adapters[f'self_attn.{p}_proj'] for p in ('q', 'v', 'o')}}
            tensors,specs=flatten_records(weights)
            self.blocks.append(BlockPlan(index, 'full_attention', dense_block, kwargs, tensors, adapters,
                                          specs if quantization!='none' else None))
        if self.storage_metadata is not None:
            self.storage_metadata.update(quantized_matrices=loader.quantized_tensors,
                host_loading='direct_packed_pinned_decoder_and_mapped_source_embedding' if direct_pinned else 'pageable_then_incremental_pin',
                direct_pinned_capacity_bytes=sum(loader.pinned_capacities),
                direct_pinned_arena_limit_bytes=512*1024**2 if direct_pinned else None,
                decoder_stored_bytes=sum(t.numel()*t.element_size() for b in self.blocks for t in b.host_weights.values()),
                cpu_embedding_bytes=self.embedding.numel()*self.embedding.element_size(),
                device_boundary_bytes=self.head.numel()*self.head.element_size()+self.norm.numel()*self.norm.element_size(),
                resident_decoder_core_bytes=0)

    def positions_and_masks(self, valid):
        if valid.dtype != torch.bool or valid.ndim != 2 or valid.shape[1] < 1:
            raise ValueError('unsupported_capability: mask must be 2D bool')
        if valid.shape[1] > self.config.max_position_embeddings:
            raise ValueError('unsupported_capability: sequence beyond configured context')
        if not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError('unsupported_capability: only all-valid or right-padding masks')
        valid = valid.to(self.device)
        batch, length = valid.shape
        positions = torch.arange(length, device=self.device).expand(batch, -1)
        inv = 1. / self.config.rope_parameters['rope_theta'] ** (torch.arange(0, self.config.head_dim, 2, device=self.device).float() / self.config.head_dim)
        freq = (inv[None, :, None].expand(batch, -1, 1) @ positions[:, None].float()).transpose(1, 2)
        phase = torch.cat((freq, freq), -1)
        rotary = (phase.cos().to(self.head.dtype), phase.sin().to(self.head.dtype))
        blocked = torch.ones(length, length, dtype=torch.bool, device=self.device).triu(1)[None, None] | ~valid[:, None, None, :]
        mask = torch.zeros(blocked.shape, dtype=self.head.dtype, device=self.device).masked_fill(blocked, torch.finfo(self.head.dtype).min)
        return rotary, {'full_attention': mask}

    def terminal_norm(self, hidden):
        return rms(hidden, self.norm, self.config.rms_norm_eps)

    def artifact_key(self, layer, target, parameter):
        return f'base_model.model.model.layers.{layer}.{target}.lora_{parameter}.weight'

    def runtime_metadata(self):
        return {'architecture': self.family, 'layer_count': len(self.blocks), 'shared_weights': 'frozen_embedding_head_source_alias' if self.source_aliases else 'untied',
                'source_aliases':self.source_aliases,'boundary_storage':'same CPU storage' if self.device.type=='cpu' and self.source_aliases else 'CPU embedding plus device head replica',
                'cpu_embedding_bytes':self.embedding.numel()*self.embedding.element_size(),'device_head_bytes':self.head.numel()*self.head.element_size(),
                'asynchronous_transfers': 'unsupported', 'kv_cache': 'disabled', 'rope': 'default_full',
                'attention': 'eager_GQA', 'positions': 'absolute_zero_start',
                'resolved_layer_types':getattr(self.config,'layer_types',['full_attention']*len(self.blocks)),
                'resolved_sliding_window':getattr(self.config,'sliding_window',None),'resolved_head_dim':self.config.head_dim}

    def close(self):
        self.blocks.clear()
        self.embedding = self.head = self.norm = None


class LlamaAdapter(DenseDecoderAdapter):
    family = 'llama'


class Qwen2Adapter(DenseDecoderAdapter):
    family = 'qwen2'


class Qwen3Adapter(DenseDecoderAdapter):
    family = 'qwen3'
