"""Qwen3.5 text dense adapter, extracted from AllTheAIs real_runtime.

Layer count is read from the checkpoint; all declared layers are trained.
No remote code, model downloads, inference offload or synthetic fallback.
"""
from ...quantization.codecs import FORMATS

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from ...finetune.lora import create_lora_for_block
from ..targets import ATTENTION_TARGETS
from ..protocol import BlockPlan
from ...quantization.storage import scan_entries,WeightLoader,estimated_bytes
from ...quantization.codecs import materialize,flatten_records
from .blocks import (functional_full_attention_block_v2, functional_linear_attention_block_v2,
                     prepare_linear_attention_core, rms_norm)

SUFFIXES = {
    "self_attn.q_proj.weight": "q_proj_weight", "self_attn.k_proj.weight": "k_proj_weight",
    "self_attn.v_proj.weight": "v_proj_weight", "self_attn.o_proj.weight": "o_proj_weight",
    "self_attn.q_norm.weight": "q_norm_weight", "self_attn.k_norm.weight": "k_norm_weight",
    "mlp.gate_proj.weight": "gate_proj_weight", "mlp.up_proj.weight": "up_proj_weight",
    "mlp.down_proj.weight": "down_proj_weight", "input_layernorm.weight": "input_layernorm_weight",
    "post_attention_layernorm.weight": "post_attention_layernorm_weight",
    "linear_attn.out_proj.weight": "out_proj_weight", "linear_attn.in_proj_qkv.weight": "in_proj_qkv_weight",
    "linear_attn.in_proj_a.weight": "in_proj_a_weight", "linear_attn.in_proj_b.weight": "in_proj_b_weight",
    "linear_attn.in_proj_z.weight": "in_proj_z_weight", "linear_attn.conv1d.weight": "conv1d_weight",
    "linear_attn.dt_bias": "dt_bias", "linear_attn.A_log": "A_log", "linear_attn.norm.weight": "norm_weight",
}


def inspect_model(path):
    root = Path(path)
    raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if raw.get("model_type") not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError("unsupported_architecture: expected dense Qwen3.5 text")
    text = raw.get("text_config", raw)
    if text.get("model_type", "qwen3_5_text") != "qwen3_5_text":
        raise ValueError("unsupported_architecture: text configuration")
    for config in (raw, text):
        if 'tie_word_embeddings' in config and type(config['tie_word_embeddings']) is not bool:
            raise ValueError('unsupported_capability: tie_word_embeddings must be boolean')
    if text is not raw and 'tie_word_embeddings' in raw and 'tie_word_embeddings' in text and raw['tie_word_embeddings'] != text['tie_word_embeddings']:
        raise ValueError('unsupported_capability: conflicting tied embedding declarations')
    tied = text.get('tie_word_embeddings', raw.get('tie_word_embeddings', False))
    # Preserve historical untied source identities (including omitted defaults).
    # Only the newly supported top-level-only tied case needs resolution here.
    if tied and 'tie_word_embeddings' not in text:
        text = {**text, 'tie_word_embeddings': True}
    if text.get("attention_dropout", 0) != 0 or text.get("attention_bias", False):
        raise ValueError("unsupported_capability: attention dropout or bias")
    if text.get("hidden_act", "silu") != "silu" or text.get("rms_norm_eps", 1e-6) != 1e-6:
        raise ValueError("unsupported_capability: nonstandard activation/norm epsilon")
    count = text["num_hidden_layers"]
    if type(count) is not int or count <= 0:
        raise ValueError("invalid layer count")
    types = text.get("layer_types")
    if types is None:
        interval = text.get("full_attention_interval", 4)
        if type(interval) is not int or interval < 1:
            raise ValueError("invalid attention interval")
        types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(count)]
    if len(types) != count or set(types) - {"full_attention", "linear_attention"}:
        raise ValueError("unsupported_capability: layer schedule")
    entries = scan_entries(root,'qwen35')
    prefix = "model.language_model."
    if prefix + "embed_tokens.weight" not in entries:
        raise ValueError("unsupported_capability: checkpoint tensor prefix")
    if tied and 'lm_head.weight' in entries:
        raise ValueError('unsupported_capability: tied source requires canonical embedding only; duplicate lm_head')
    required = {prefix + "embed_tokens.weight", prefix + "norm.weight"}
    if not tied:
        required.add('lm_head.weight')
    for i, kind in enumerate(types):
        required.update(prefix + f"layers.{i}." + suffix for suffix in SUFFIXES
                        if not (suffix.startswith("self_attn.") and kind != "full_attention")
                        and not (suffix.startswith("linear_attn.") and kind != "linear_attention"))
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required tensors: {sorted(missing)}")
    hidden, vocab = text.get('hidden_size'), text.get('vocab_size')
    if type(hidden) is not int or type(vocab) is not int or min(hidden, vocab) < 1:
        raise ValueError('invalid hidden/vocabulary dimensions')
    boundaries = {prefix+'embed_tokens.weight': [vocab, hidden], prefix+'norm.weight': [hidden]}
    if not tied:
        boundaries['lm_head.weight'] = [vocab, hidden]
    if any(entries[k]['shape'] != expected for k, expected in boundaries.items()):
        raise ValueError('checkpoint/config boundary shape mismatch')
    decoder_keys = {k for k in entries if k.startswith(prefix + "layers.")}
    if decoder_keys - required:
        raise ValueError(f"unsupported_capability: unexpected decoder tensors {sorted(decoder_keys - required)}")
    excluded = set(entries) - required
    if any(not k.startswith(('model.visual.', 'mtp.')) for k in excluded):
        raise ValueError('unsupported_capability: unknown tensors outside declared text-only scope')
    if any(entries[k]["dtype"] not in {"F32", "BF16"} for k in required):
        raise ValueError("unsupported_capability: only FP32/BF16 frozen weights")
    def size(key):
        import math
        return entries[key].get('stored_bytes',math.prod(entries[key]["shape"]) * (4 if entries[key]["dtype"] == "F32" else 2))
    host_bytes = sum(size(k) for k in required if k not in {"lm_head.weight", prefix + "norm.weight"})
    return {"quantized_export": {"version":1,"formats":[] if tied else list(FORMATS),"rejection_reason":"tied Qwen3.5 compressed path not validated" if tied else ""}, "architecture": "qwen35", "layers": count, "layer_types": types,
            "text_config": text, "entries": entries, "required": sorted(required),
            'quantization_matrix_keys':[k for k in required if k.startswith(prefix+'layers.') and len(entries[k]['shape'])==2 and math.prod(entries[k]['shape'])>=128],
            'quantization_fp32_keys':[k for k in required if k.endswith(('.dt_bias','.A_log'))],
            'source_aliases': {'lm_head.weight': prefix+'embed_tokens.weight'} if tied else {},
            'training_scope': 'text causal cross-entropy only; vision and MTP excluded',
            'excluded_nontext_tensors': {role: sorted(k for k in excluded if k.startswith(start))
                for role,start in [('vision','model.visual.'),('mtp','mtp.')]},
            "host_weight_bytes": host_bytes, "head_bytes": size(prefix+'embed_tokens.weight' if tied else 'lm_head.weight')}


class Qwen35Adapter:
    def __init__(self, path, *, rank=2, alpha=4.0, device="cpu", host_budget_bytes=1024**3,
                 device_budget_bytes=2*1024**3,quantization='none',quantized_compute='bf16'):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
        report = inspect_model(path)
        if quantized_compute!='bf16':raise ValueError('unsupported_quantization: only storage plus BF16 compute')
        if report['source_aliases'] and quantization != 'none':
            raise ValueError('unsupported_quantization: tied Qwen3.5 compressed path not validated')
        direct_pinned = quantization=='int4-sym-g128' and torch.device(device).type=='cuda'
        loader=WeightLoader(path,report,quantization,
            pin_keys=[k for k in report['required'] if k.startswith('model.language_model.layers.')] if direct_pinned else (),
            host_budget_bytes=host_budget_bytes)
        boundaries={'lm_head.weight','model.language_model.norm.weight'}
        head_key = report['source_aliases'].get('lm_head.weight','lm_head.weight')
        if sum(estimated_bytes(report,quantization,k) for k in report['required'] if k not in boundaries) > host_budget_bytes or estimated_bytes(report,quantization,head_key) > device_budget_bytes:
            raise MemoryError("resource_budget: model exceeds host/head allowance before loading")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("dependency: explicit CUDA requested but unavailable")
        self.config = Qwen3_5TextConfig(**report["text_config"])
        self.config._attn_implementation = "eager"
        state = {}
        for key in report["required"]:
            record=loader.read(key)
            state[key]=materialize(record,self.device) if key in boundaries else record
        self.identity=loader.finish()
        self.storage_metadata={'format':quantization,'compute':'bf16' if quantization!='none' else 'source',
            'storage_identity':loader.storage_identity,'source_identity':loader.source_identity,
            'total_stored_bytes':loader.stored_bytes,'max_source_tensor_bytes':loader.max_source_tensor_bytes,
            'full_bf16_decoder_retained':False,'quantized_scope':'decoder matrices >=128 elements; boundaries/norm/bias/conv excluded'} if quantization!='none' else None
        prefix = "model.language_model."
        self.embedding = materialize(state.pop(prefix + "embed_tokens.weight"))
        self.norm = state.pop(prefix + "norm.weight").to(self.device)
        self.source_aliases = report['source_aliases']
        self.training_scope = report['training_scope']
        self.excluded_nontext_tensors = report['excluded_nontext_tensors']
        # The canonical frozen tensor is read once. CPU aliases storage;
        # CUDA keeps an explicit immutable head replica while embedding stays RAM.
        self.head = self.embedding.to(self.device) if self.source_aliases else state.pop('lm_head.weight').to(self.device)
        if self.head.dtype != self.embedding.dtype or self.norm.dtype != self.head.dtype:
            raise ValueError("unsupported_capability: mixed boundary dtypes")
        self.rotary = Qwen3_5TextRotaryEmbedding(self.config).to(self.device).eval()
        self.blocks, self.cores = [], []
        shared_core = None
        for index, kind in enumerate(report["layer_types"]):
            layer_prefix = prefix + f"layers.{index}."
            layer_state = {k.removeprefix(layer_prefix): v for k, v in state.items() if k.startswith(layer_prefix)}
            with torch.device("meta"):
                expected_layer = Qwen3_5DecoderLayer(self.config, index)
            expected = expected_layer.state_dict()
            if set(expected) != set(layer_state) or any(expected[k].shape != v.shape for k, v in layer_state.items()):
                raise ValueError(f"checkpoint/config layer shape mismatch at layer {index}")
            del expected_layer, expected
            if any(k not in SUFFIXES for k in layer_state):
                raise ValueError("unsupported decoder tensor")
            weights = {SUFFIXES[k]: v for k, v in sorted(layer_state.items())}
            adapters = create_lora_for_block(index, kind, {k:v.shape for k,v in layer_state.items()}, rank=rank, alpha=alpha,
                                             device=self.device, targets=ATTENTION_TARGETS[kind], architecture="Qwen3.5")
            if kind == "linear_attention":
                if shared_core is None:
                    with torch.device("meta"):
                        layer = Qwen3_5DecoderLayer(self.config, index)
                    core=layer.linear_attn
                    core.load_state_dict({k.removeprefix('linear_attn.'):materialize(v) for k,v in layer_state.items() if k.startswith('linear_attn.')},strict=True,assign=True)
                    core=core.to(self.device).requires_grad_(False).eval()
                    shared_core, _ = prepare_linear_attention_core(core)
                    self.cores.append(shared_core)
                    del layer
                functional = functional_linear_attention_block_v2
                kwargs = {"hf_linear_attn_core": shared_core, "eps": self.config.rms_norm_eps,
                          "lora_out": adapters["linear_attn.out_proj"]}
            else:
                functional = functional_full_attention_block_v2
                kwargs = {"num_heads": self.config.num_attention_heads,
                          "num_kv_heads": self.config.num_key_value_heads,
                          "head_dim": self.config.head_dim, "eps": self.config.rms_norm_eps,
                          **{f"lora_{p}": adapters[f"self_attn.{p}_proj"] for p in ("q", "v", "o")}}
            tensors,specs=flatten_records(weights)
            self.blocks.append(BlockPlan(index, kind, functional, kwargs, tensors, adapters,
                                          specs if quantization!='none' else None))
        if self.storage_metadata is not None:
            self.storage_metadata.update(quantized_matrices=loader.quantized_tensors,
                host_loading='direct_packed_pinned_decoder_and_mapped_source_embedding' if direct_pinned else 'pageable_then_incremental_pin',
                direct_pinned_capacity_bytes=sum(loader.pinned_capacities),
                decoder_stored_bytes=sum(t.numel()*t.element_size() for b in self.blocks for t in b.host_weights.values()),
                cpu_embedding_bytes=self.embedding.numel()*self.embedding.element_size(),
                device_boundary_bytes=self.head.numel()*self.head.element_size()+self.norm.numel()*self.norm.element_size(),
                resident_decoder_core_bytes=sum(p.numel()*p.element_size() for c in self.cores for p in c.parameters()))

    def positions_and_masks(self, valid):
        valid = valid.to(self.device)
        batch, length = valid.shape
        positions = torch.arange(length, device=self.device).expand(batch, -1)
        with torch.no_grad():
            pos = self.rotary(torch.empty((), dtype=self.head.dtype, device=self.device), positions)
        blocked = torch.triu(torch.ones(length, length, device=self.device, dtype=torch.bool), 1)
        blocked = blocked[None, None].expand(batch, 1, length, length) | ~valid[:, None, None, :]
        full = torch.zeros(blocked.shape, dtype=self.head.dtype, device=self.device)
        full.masked_fill_(blocked, torch.finfo(self.head.dtype).min)
        return pos, {"full_attention": full, "linear_attention": valid}

    def terminal_norm(self, hidden):
        return rms_norm(hidden, self.norm, self.config.rms_norm_eps)

    def artifact_key(self, layer, target, parameter):
        return f"base_model.model.model.language_model.layers.{layer}.{target}.lora_{parameter}.weight"

    def runtime_metadata(self):
        return {"architecture": "qwen3_5_text", "layer_count": len(self.blocks),
                "linear_backends": sorted({core.chunk_gated_delta_rule.__module__ + "." + core.chunk_gated_delta_rule.__name__
                                           for core in self.cores}),
                "shared_weights": 'frozen embedding/head alias only' if self.source_aliases else 'none',
                'source_aliases': self.source_aliases, 'training_scope': self.training_scope,
                'excluded_nontext_tensor_counts': {k:len(v) for k,v in self.excluded_nontext_tensors.items()},
                'cpu_embedding_bytes': self.embedding.numel()*self.embedding.element_size(),
                'device_head_bytes': self.head.numel()*self.head.element_size(),
                'head_replica': 'frozen CUDA replica' if self.source_aliases and self.device.type=='cuda' else 'CPU alias' if self.source_aliases else 'separate frozen head',
                "asynchronous_transfers": "unsupported"}

    def close(self):
        self.blocks.clear(); self.cores.clear()
        self.embedding = self.norm = self.head = self.rotary = None
