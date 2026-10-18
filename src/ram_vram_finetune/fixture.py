"""Generate a tiny real HF architecture with random weights, never download."""
import json
from pathlib import Path


def make_tiny(output, *, layers=4, adapter='qwen35'):
    if adapter=='qwen35_moe':
        from .qwen35_moe_fixture import make_tiny as native_tiny
        if type(layers) is not int or layers<1:raise ValueError('layers must be a positive integer')
        return native_tiny(output,layers=layers)
    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    output = Path(output)
    if type(layers) is not int or layers < 1:
        raise ValueError("layers must be a positive integer")
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(731)
    if adapter not in {'qwen35', 'llama', 'qwen2','qwen3','qwen3_moe','deepseek_v3','gpt_oss','glm4_moe_lite'}:
        raise ValueError('unsupported_architecture: tiny fixture adapter')
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_conv_kernel_dim=3,
        layer_types=["linear_attention" if i % 2 == 0 else "full_attention" for i in range(layers)],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
        attention_dropout=0.0, use_cache=False, tie_word_embeddings=False)
    if adapter == 'qwen35':
        config._attn_implementation = 'eager'
        model = Qwen3_5TextModel(config).float().eval()
        state = {"model.language_model." + k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
        state["lm_head.weight"] = torch.randn(config.vocab_size, config.hidden_size) * 0.1
    elif adapter=='qwen3_moe':
        from .qwen3_moe_fixture import state as moe_state
        config,state=moe_state(layers)
    elif adapter=='glm4_moe_lite':
        from .glm_fixture import state as glm_state
        config,state=glm_state(layers)
    elif adapter=='gpt_oss':
        from .adapters.gpt_oss.fixture import state as gptoss_state
        config,state=gptoss_state(layers)
    elif adapter=='deepseek_v3':
        from .deepseek_fixture import state as deepseek_state
        config,state=deepseek_state(layers)
    else:
        from .adapters.dense import hf_classes
        Config, _, Model = hf_classes(adapter)
        config = Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
                        attention_dropout=0., tie_word_embeddings=(adapter=='qwen3'), use_cache=False,
                        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.})
        config._attn_implementation = 'eager'
        model = Model(config).float().eval()
        state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
        if adapter=='qwen3':
            state.pop('lm_head.weight')
            for key,value in state.items():
                if key.endswith(('q_norm.weight','k_norm.weight')):value.uniform_(.7,1.3)
        if adapter == 'qwen2':
            # HF initialization uses zero biases. Nonzero frozen biases ensure
            # parity tests actually exercise Qwen2's projection distinction.
            for key, value in state.items():
                if key.endswith(('.q_proj.bias', '.k_proj.bias', '.v_proj.bias')):
                    value.copy_(torch.randn_like(value) * .1)
    save_file(state, str(output / "model.safetensors"))
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8", newline="\n")
    rows = [{"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
            {"input_ids": [2, 4, 6, 8], "labels": [-100, 4, 6, 8]},
            {"input_ids": [3, 6, 9], "labels": [-100, -100, 9]}]
    (output / "tokens.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {token: i for i, token in enumerate(("<pad>", "<unk>", "<eos>", "user", "assistant",
                                                ":", "hello", "world", "answer"))}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>",
                                  eos_token="<eos>", pad_token="<pad>")
    if adapter in {'qwen2','qwen3','qwen3_moe'}:
        # HF 5.13 AutoTokenizer selects Qwen2's BPE implementation by model
        # type even for a generic TokenizersBackend declaration. Use a real
        # tiny Qwen2 BPE so that reloading never reinterprets WordLevel data.
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        alphabet = sorted(set('user:helloworldassistantanswer') | {'Ġ'})
        bpe_vocab = {token: i for i, token in enumerate(['<pad>', '<unk>', '<eos>', *alphabet])}
        fast = Qwen2Tokenizer(vocab=bpe_vocab, merges=[], unk_token='<unk>', eos_token='<eos>', pad_token='<pad>')
    fast.chat_template = "{% for message in messages %}{{ message['role'] }} : {{ message['content'] }} <eos> {% endfor %}"
    fast.save_pretrained(output)
    (output / "chat.jsonl").write_text(json.dumps({"sample_id": "tiny-chat-1", "messages": [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "answer world"}]}) + "\n", encoding="utf-8", newline="\n")
    return output
