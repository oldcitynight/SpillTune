"""Small packed random native source; all ordinary biases/sinks are nonzero."""
import torch
from .adapter import hf_classes


def state(layers=2,*,dtype=torch.float32,top_k=2,intermediate=64):
    Config,_,Model=hf_classes()
    cfg=Config(vocab_size=32,hidden_size=32,intermediate_size=intermediate,num_hidden_layers=layers,num_attention_heads=4,num_key_value_heads=2,head_dim=16,num_local_experts=4,num_experts_per_tok=top_k,
        layer_types=['sliding_attention' if i%2==0 else 'full_attention' for i in range(layers)],sliding_window=2,max_position_embeddings=128,
        attention_bias=True,attention_dropout=0.,tie_word_embeddings=False,use_cache=False,output_router_logits=False,router_aux_loss_coef=.9,
        rope_parameters={'rope_type':'yarn','rope_theta':150000.,'factor':4.,'original_max_position_embeddings':32,'beta_fast':32.,'beta_slow':1.,'truncate':False})
    cfg._attn_implementation='eager';cfg._experts_implementation='eager'
    with torch.device('meta'):shapes={k:list(v.shape) for k,v in Model(cfg).state_dict().items()}
    values={}
    for key,shape in shapes.items():
        if key.endswith(('experts.gate_up_proj','experts.down_proj')):
            e,inp,out=shape;values[key+'_blocks']=torch.randint(0,256,(e,out,inp//32,16),dtype=torch.uint8)
            values[key+'_scales']=torch.randint(118,121,(e,out,inp//32),dtype=torch.uint8)
        else:
            value=torch.randn(shape)*(.1 if key.endswith(('.bias','_bias','sinks')) else .05)
            if 'layernorm.weight' in key or key=='model.norm.weight':value=value+1.
            values[key]=value.to(dtype)
    cfg.quantization_config={'quant_method':'mxfp4','modules_to_not_convert':['model.layers.*.self_attn','model.layers.*.mlp.router','model.embed_tokens','lm_head']}
    return cfg,values
