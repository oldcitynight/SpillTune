"""Small real native DeepSeek-V3 topology; random, never pretrained/downloaded."""
import torch
from .adapters.deepseek import hf_classes


def state(layers,*,layout='packed',q_rank=8,rope='yarn',interleave=True,dtype=torch.float32):
    if layout not in {'packed','individual'} or rope not in {'default','yarn'}:raise ValueError('invalid DeepSeek fixture variant')
    Config,_,Model=hf_classes();torch.manual_seed(731)
    rotary={'rope_type':rope,'rope_theta':10000.}
    if rope=='yarn':rotary.update(factor=4.,original_max_position_embeddings=16,beta_fast=4.,beta_slow=1.,mscale=1.,mscale_all_dim=1.)
    c=Config(vocab_size=32,hidden_size=16,intermediate_size=32,moe_intermediate_size=16,num_hidden_layers=layers,
        num_attention_heads=2,num_key_value_heads=2,kv_lora_rank=8,q_lora_rank=q_rank,qk_nope_head_dim=4,
        qk_rope_head_dim=4,v_head_dim=4,n_routed_experts=4,n_shared_experts=2,n_group=2,topk_group=1,
        num_experts_per_tok=2,first_k_dense_replace=min(1,layers-1),norm_topk_prob=True,routed_scaling_factor=1.7,
        max_position_embeddings=64,rms_norm_eps=3e-5,rope_parameters=rotary,rope_interleave=interleave,
        attention_dropout=0.,attention_bias=False,tie_word_embeddings=False,use_cache=False)
    c._attn_implementation='eager';c._experts_implementation='eager';model=Model(c).eval()
    with torch.no_grad():
        for layer in model.model.layers[c.first_k_dense_replace:]:
            layer.mlp.gate.weight.normal_(0,.3)
            layer.mlp.gate.e_score_correction_bias.copy_(torch.tensor([.03,-.02,.01,-.015]))
    weights={k:v.detach().to(dtype).clone().contiguous() for k,v in model.state_dict().items()}
    # Native mixed BF16 checkpoint policy: routing remains F32.
    for k in weights:
        if k.endswith(('.mlp.gate.weight','.mlp.gate.e_score_correction_bias')):weights[k]=weights[k].float()
    if layout=='individual':
        for i in range(c.first_k_dense_replace,layers):
            prefix=f'model.layers.{i}.mlp.experts.';gu=weights.pop(prefix+'gate_up_proj');down=weights.pop(prefix+'down_proj')
            for e in range(c.n_routed_experts):
                gate,up=gu[e].chunk(2,0)
                weights.update({prefix+f'{e}.gate_proj.weight':gate.clone().contiguous(),prefix+f'{e}.up_proj.weight':up.clone().contiguous(),
                                prefix+f'{e}.down_proj.weight':down[e].clone().contiguous()})
    return c,weights
