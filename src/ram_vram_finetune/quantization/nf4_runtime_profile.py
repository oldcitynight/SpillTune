"""Explicit, source-bound model profiles; a profile is not GPU qualification."""
from __future__ import annotations
import hashlib,json,math,re
from collections.abc import Mapping

SCHEMA='rvft.nf4.runtime_profile.v1'
TARGETS={'linear_attention':('linear_attn.out_proj',),'full_attention':('self_attn.q_proj','self_attn.v_proj','self_attn.o_proj')}

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def _digest(value,label):
    if type(value)is not str or re.fullmatch('[0-9a-f]{64}',value)is None:
        raise ValueError('invalid '+label)
    return value

def validate_text_config(config):
    if not isinstance(config,Mapping):raise ValueError('resolved text config must be a mapping')
    c=json.loads(canonical(dict(config)))
    if c.get('model_type')!='qwen3_5_text' or c.get('tie_word_embeddings') is not True:
        raise ValueError('explicit profile supports tied Qwen3.5 text only')
    for key in ('hidden_size','intermediate_size','vocab_size','num_hidden_layers'):
        if type(c.get(key))is not int or c[key]<=0:raise ValueError('invalid config '+key)
    kinds=c.get('layer_types')
    if type(kinds)is not list or len(kinds)!=c['num_hidden_layers'] or any(type(k)is not str or k not in TARGETS for k in kinds):
        raise ValueError('exact per-layer attention layout is required')
    if c.get('attention_dropout',0)!=0 or c.get('attention_bias',False)is not False:
        raise ValueError('unsupported dropout/bias policy')
    if c.get('hidden_act','silu')!='silu' or c.get('rms_norm_eps',1e-6)!=1e-6:
        raise ValueError('unsupported activation/norm policy')
    return c

def target_names(config):
    c=validate_text_config(config)
    return sorted(f'{i}:{name}' for i,kind in enumerate(c['layer_types']) for name in TARGETS[kind])

def make_runtime_profile(config,artifact_identity,source_identity,rank=8,alpha=16,seed=20261019):
    c=validate_text_config(config)
    _digest(artifact_identity,'artifact identity');_digest(source_identity,'source payload identity')
    if type(rank)is not int or rank<=0:raise ValueError('rank must be a positive integer')
    if type(alpha)not in (int,float) or not math.isfinite(alpha) or alpha<=0:raise ValueError('alpha must be positive finite')
    if type(seed)is not int or seed<0:raise ValueError('seed must be a nonnegative integer')
    value={'schema':SCHEMA,'architecture':'qwen35','config_sha256':hashlib.sha256(canonical(c)).hexdigest(),
      'artifact_identity':artifact_identity,'source_identity':source_identity,'rank':rank,'alpha':float(alpha),
      'seed':seed,'targets':target_names(c),'layer_types':c['layer_types'],'tie_word_embeddings':True,
      'quantization':{'format':'official NF4','block_size':64,'double_quantization':True,'double_quant_block_size':256,'compute_dtype':'bfloat16'},
      'qualification_claim':'none; explicit construction contract only'}
    value['profile_sha256']=hashlib.sha256(canonical(value)).hexdigest()
    return value

def validate_runtime_profile(profile,config,artifact_identity,source_identity):
    if not isinstance(profile,Mapping):raise ValueError('explicit runtime profile required')
    p=json.loads(canonical(dict(profile)))
    expected=make_runtime_profile(config,artifact_identity,source_identity,p.get('rank'),p.get('alpha'),p.get('seed'))
    # Canonical bytes distinguish bools/integers and forbid extra or omitted fields.
    if canonical(p)!=canonical(expected):raise ValueError('runtime profile does not exactly match artifact/config/source/targets')
    return p
