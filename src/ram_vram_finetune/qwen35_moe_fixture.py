"""Small native Qwen3.5-MoE text source; no pretrained data or downloads."""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file


def make_tiny(path,*,layers=4):
    from .adapters.qwen35_moe import hf_classes
    Config,h=hf_classes();torch.manual_seed(733)
    cfg=Config(vocab_size=32,hidden_size=16,num_hidden_layers=layers,num_attention_heads=2,num_key_value_heads=1,head_dim=8,num_experts=3,num_experts_per_tok=2,moe_intermediate_size=8,shared_expert_intermediate_size=8,linear_num_value_heads=2,linear_num_key_heads=1,linear_key_head_dim=8,linear_value_head_dim=8,linear_conv_kernel_dim=4,layer_types=['linear_attention']*(layers-1)+['full_attention'],max_position_embeddings=64,attention_dropout=0.,attention_bias=False,output_router_logits=False,tie_word_embeddings=False,rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5,'mrope_section':[1,1,0],'mrope_interleaved':True})
    cfg._attn_implementation='eager';cfg._experts_implementation='eager'
    model=h.Qwen3_5MoeForCausalLM(cfg).eval().requires_grad_(False).bfloat16()
    state={('model.language_model.'+k[len('model.'):]) if k.startswith('model.') else k:v.contiguous() for k,v in model.state_dict().items()}
    path=Path(path);path.mkdir(parents=True,exist_ok=False);save_file(state,str(path/'model.safetensors'));(path/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2)+'\n', encoding="utf-8", newline="\n")
    rows=[{'input_ids':[1,2,3,4,5,6,7,8],'labels':[-100,-100,3,4,5,6,7,8]},{'input_ids':[2,4,6,8,10,12],'labels':[-100,-100,6,8,10,12]}]
    (path/'tokens.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows), encoding="utf-8", newline="\n");return path
