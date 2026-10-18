# Adapted from AllTheAIs streaming/lora.py; see SOURCE_ATTRIBUTION.md.
"""Explicit adapter-owned legacy target mappings; ordering is checkpoint-stable."""
ATTENTION_TARGETS: dict[str, tuple[str, ...]] = {
    "full_attention": (
        "self_attn.q_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ),
    "linear_attention": ("linear_attn.out_proj",),
}

