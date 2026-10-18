"""Tiny random native GLM MoE Lite; independent library graph, no download."""
import torch
from .adapters.glm4_moe_lite import hf_classes


def state(layers,*,layout='individual',dtype=torch.float32,mtp=False):
    if layers<2 or layout not in {'packed','individual'}:raise ValueError('GLM fixture layers/layout')
    Config,_,Model=hf_classes();torch.manual_seed(731)
    c=Config(vocab_size=32,hidden_size=16,intermediate_size=32,moe_intermediate_size=16,num_hidden_layers=layers,num_attention_heads=2,num_key_value_heads=2,kv_lora_rank=8,q_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=6,n_routed_experts=4,n_shared_experts=1,n_group=2,topk_group=1,num_experts_per_tok=2,norm_topk_prob=True,routed_scaling_factor=1.8,max_position_embeddings=64,rms_norm_eps=3e-5,rope_parameters={'rope_type':'default','rope_theta':10000.},rope_interleave=True,attention_dropout=0.,attention_bias=False,tie_word_embeddings=False,use_cache=False)
    c._attn_implementation='eager';c._experts_implementation='eager';c.first_k_dense_replace=1;c.num_nextn_predict_layers=int(mtp)
    model=Model(c).eval()
    with torch.no_grad():
        for layer in model.model.layers[1:]:layer.mlp.gate.weight.normal_(0,.3);layer.mlp.gate.e_score_correction_bias.copy_(torch.tensor([.03,-.02,.01,-.015]))
    weights={k:v.detach().to(dtype).clone().contiguous() for k,v in model.state_dict().items()}
    for k in weights:
        if k.endswith(('.mlp.gate.weight','.mlp.gate.e_score_correction_bias')):weights[k]=weights[k].float()
    if layout=='individual':
        for i in range(1,layers):
            prefix=f'model.layers.{i}.mlp.experts.';gu=weights.pop(prefix+'gate_up_proj');down=weights.pop(prefix+'down_proj')
            for e in range(c.n_routed_experts):
                gate,up=gu[e].chunk(2,0)
                for name,value in [('gate_proj',gate),('up_proj',up),('down_proj',down[e])]:weights[prefix+f'{e}.{name}.weight']=value.clone().contiguous()
    if mtp:
        prefix=f'model.layers.{layers}.';last=f'model.layers.{layers-1}.'
        weights.update({prefix+k.removeprefix(last):v.clone() for k,v in list(weights.items()) if k.startswith(last)})
        for name,shape in [('embed_tokens.weight',(32,16)),('shared_head.head.weight',(32,16)),('shared_head.norm.weight',(16,)),('eh_proj.weight',(16,32)),('enorm.weight',(16,)),('hnorm.weight',(16,))]:weights[prefix+name]=(torch.randn(shape)*.02).to(dtype).contiguous()
    return c,weights
