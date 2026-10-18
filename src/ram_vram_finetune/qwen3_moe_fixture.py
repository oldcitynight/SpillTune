"""Random native HF Qwen3-MoE fixture, never pretrained source weights."""
import torch


def state(layers=2,*,layout='per_expert_2d',experts=3,top_k=1,norm_topk=True,tied=True):
    from .adapters.qwen3_moe import hf_classes
    Config,_,Model=hf_classes()
    config=Config(vocab_size=32,hidden_size=16,intermediate_size=32,moe_intermediate_size=32,
        num_hidden_layers=layers,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        max_position_embeddings=64,num_experts=experts,num_experts_per_tok=top_k,
        norm_topk_prob=norm_topk,decoder_sparse_step=1,mlp_only_layers=[],tie_word_embeddings=tied,
        attention_dropout=0.,attention_bias=False,use_sliding_window=False,sliding_window=None,
        output_router_logits=False,router_aux_loss_coef=.001,use_cache=False,
        rope_parameters={'rope_type':'default','rope_theta':10000.})
    config._attn_implementation='eager';config._experts_implementation='eager';model=Model(config).float().eval()
    result={k:v.detach().clone().contiguous() for k,v in model.state_dict().items()}
    if tied:result.pop('lm_head.weight')
    for key,value in result.items():
        if key.endswith(('q_norm.weight','k_norm.weight')):value.uniform_(.7,1.3)
        elif key.endswith('mlp.gate.weight'):value.normal_(0,.5)
    if layout=='per_expert_2d':
        for i in range(layers):
            prefix=f'model.layers.{i}.mlp.experts.';gu=result.pop(prefix+'gate_up_proj');down=result.pop(prefix+'down_proj')
            for e in range(experts):
                gate,up=gu[e].chunk(2,dim=0)
                for name,tensor in [('gate_proj',gate),('up_proj',up),('down_proj',down[e])]:result[f'{prefix}{e}.{name}.weight']=tensor.clone().contiguous()
    elif layout!='packed_3d':raise ValueError('fixture expert layout')
    return config,result
