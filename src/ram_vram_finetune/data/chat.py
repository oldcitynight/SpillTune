# Adapted from AllTheAIs backend/train/data.py; see SOURCE_ATTRIBUTION.md.
"""Strict JSONL/chat tokenization for real causal-LM training.

The Qwen tokenizer used by the real training workload has no Jinja
``{% generation %}`` blocks, so ``return_assistant_tokens_mask=True`` cannot
identify assistant tokens.  This module instead locates assistant *body*
character spans in the rendered conversation and intersects those spans with
the fast-tokenizer offset mapping.

No function in this module truncates input.  A sample that exceeds the caller's
maximum sequence length is rejected with its original token count intact.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TokenizationMetadata:
    """Auditable metadata for one rendered JSONL conversation."""

    sample_id: str
    total_token_count: int
    supervised_token_count: int
    first_supervised_token_index: int
    last_supervised_token_index: int
    rendered_text_sha256: str
    assistant_text_sha256: str
    tokenization_time_s: float
    assistant_char_spans: tuple[tuple[int, int], ...]
    assistant_text_sha256s: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record suitable for an inventory file."""

        value = asdict(self)
        value["assistant_char_spans"] = [list(span) for span in self.assistant_char_spans]
        value["assistant_text_sha256s"] = list(self.assistant_text_sha256s)
        return value


@dataclass(frozen=True)
class TokenizedSample:
    """One unpadded causal-LM sample on CPU."""

    sample_id: str
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    assistant_mask: torch.Tensor
    offset_mapping: tuple[tuple[int, int], ...]
    rendered_text: str
    metadata: TokenizationMetadata

    def __post_init__(self) -> None:
        length = self.input_ids.numel()
        if self.input_ids.ndim != 1 or self.input_ids.dtype != torch.long:
            raise TypeError("input_ids must be a one-dimensional torch.long tensor")
        if self.labels.shape != self.input_ids.shape or self.labels.dtype != torch.long:
            raise TypeError("labels must be torch.long and match input_ids")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if (
            self.assistant_mask.shape != self.input_ids.shape
            or self.assistant_mask.dtype != torch.bool
        ):
            raise TypeError("assistant_mask must be torch.bool and match input_ids")
        if len(self.offset_mapping) != length:
            raise ValueError("offset_mapping must have one entry per input token")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_real_tokenizer(model_path: str | Path):
    """Load the required fast tokenizer with the locked regex correction."""

    from transformers import AutoTokenizer

    # This is an explicitly supplied local model directory; local_files_only
    # prevents a Hub lookup, so a remote revision pin is not applicable.
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        str(model_path),
        trust_remote_code=False,
        fix_mistral_regex=True,
        local_files_only=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("real causal-LM training requires a fast tokenizer")
    return tokenizer


def _validate_messages(messages: object) -> list[dict[str, Any]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("messages must be a sequence of mappings")
    if not messages:
        raise ValueError("messages must not be empty")

    normalized: list[dict[str, Any]] = []
    assistant_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"messages[{index}] must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise TypeError(f"messages[{index}].role must be a non-empty string")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}].role is unsupported: {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"messages[{index}].content must be a string")
        if role == "assistant":
            assistant_count += 1
            if not content:
                raise ValueError(f"messages[{index}] has an empty assistant body")
        normalized.append(dict(message))

    if assistant_count == 0:
        raise ValueError("conversation contains no assistant message")
    return normalized


def _render(tokenizer, messages: Sequence[Mapping[str, Any]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template(..., tokenize=False) must return str")
    return rendered


def _assistant_body_spans(
    tokenizer,
    messages: Sequence[Mapping[str, Any]],
    rendered_text: str,
) -> tuple[tuple[int, int], ...]:
    """Locate every assistant body without relying on generation masks.

    Each assistant body is wrapped with deterministic private-use markers in a
    second render.  Removing those markers must reproduce the original render
    byte-for-byte.  This both disambiguates repeated assistant/user text and
    detects templates that transform assistant content in a way that would make
    a character-offset mask unsafe.
    """

    marked_messages = [dict(message) for message in messages]
    markers: list[tuple[str, str]] = []
    assistant_ordinal = 0
    for message in marked_messages:
        if message["role"] != "assistant":
            continue
        start_marker = f"\ue000R3_ASSISTANT_{assistant_ordinal}_START\ue001"
        end_marker = f"\ue002R3_ASSISTANT_{assistant_ordinal}_END\ue003"
        content = message["content"]
        if start_marker in rendered_text or end_marker in rendered_text:
            raise ValueError("assistant span marker collides with rendered text")
        message["content"] = start_marker + content + end_marker
        markers.append((start_marker, end_marker))
        assistant_ordinal += 1

    marked_render = _render(tokenizer, marked_messages)
    clean_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    marked_cursor = 0
    clean_length = 0
    for ordinal, (start_marker, end_marker) in enumerate(markers):
        start = marked_render.find(start_marker, marked_cursor)
        if start < 0 or marked_render.find(start_marker, start + len(start_marker)) >= 0:
            raise ValueError(f"assistant start marker {ordinal} was not rendered exactly once")
        prefix = marked_render[marked_cursor:start]
        clean_parts.append(prefix)
        clean_length += len(prefix)

        body_start = start + len(start_marker)
        end = marked_render.find(end_marker, body_start)
        if end < 0 or marked_render.find(end_marker, end + len(end_marker)) >= 0:
            raise ValueError(f"assistant end marker {ordinal} was not rendered exactly once")
        body = marked_render[body_start:end]
        span_start = clean_length
        clean_parts.append(body)
        clean_length += len(body)
        spans.append((span_start, clean_length))
        marked_cursor = end + len(end_marker)

    clean_parts.append(marked_render[marked_cursor:])
    reconstructed = "".join(clean_parts)
    if reconstructed != rendered_text:
        raise ValueError(
            "chat template transforms assistant content; exact assistant-body "
            "character spans cannot be proven"
        )
    if any(start >= end for start, end in spans):
        raise ValueError("assistant body produced an empty rendered character span")
    return tuple(spans)


def tokenize_conversation(
    tokenizer,
    *,
    sample_id: str,
    messages: Sequence[Mapping[str, Any]],
    max_sequence_length: int,
) -> TokenizedSample:
    """Render and tokenize one conversation with assistant-only labels."""

    started = time.perf_counter()
    if not isinstance(sample_id, str) or not sample_id:
        raise TypeError("sample_id must be a non-empty string")
    if isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int):
        raise TypeError("max_sequence_length must be an integer")
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be >= 1")
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("assistant offset masking requires a fast tokenizer")

    normalized_messages = _validate_messages(messages)
    rendered_text = _render(tokenizer, normalized_messages)
    assistant_spans = _assistant_body_spans(tokenizer, normalized_messages, rendered_text)

    encoded = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )
    if "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise KeyError("fast tokenizer result must contain input_ids and offset_mapping")
    raw_input_ids = encoded["input_ids"]
    raw_offsets = encoded["offset_mapping"]
    if raw_input_ids and isinstance(raw_input_ids[0], (list, tuple)):
        raise ValueError("tokenizer returned a batch for one rendered conversation")
    input_ids_list = [int(value) for value in raw_input_ids]
    offsets = tuple((int(start), int(end)) for start, end in raw_offsets)
    if len(offsets) != len(input_ids_list):
        raise ValueError("offset_mapping length does not match input_ids")

    total_token_count = len(input_ids_list)
    if total_token_count > max_sequence_length:
        raise ValueError(
            f"sample {sample_id!r} has {total_token_count} tokens, exceeding "
            f"max_sequence_length={max_sequence_length}; truncation is forbidden"
        )

    rendered_length = len(rendered_text)
    assistant_mask_list: list[bool] = []
    for token_index, (token_start, token_end) in enumerate(offsets):
        if not (0 <= token_start <= token_end <= rendered_length):
            raise ValueError(
                f"invalid offset_mapping[{token_index}]={(token_start, token_end)} "
                f"for rendered length {rendered_length}"
            )
        supervised = token_end > token_start and any(
            token_start < span_end and token_end > span_start
            for span_start, span_end in assistant_spans
        )
        assistant_mask_list.append(supervised)

    supervised_indices = [
        index for index, supervised in enumerate(assistant_mask_list) if supervised
    ]
    if not supervised_indices:
        raise ValueError(f"sample {sample_id!r} produced zero supervised tokens")

    # terminal_supervision_v3_last_control
    # Qwen chat templates can append a terminal control token and then ordinary
    # formatting whitespace. Supervise through the last recognized control
    # token after the final assistant body, while leaving trailing whitespace
    # outside the loss. This teaches EOT/EOS without forcing template newlines.
    if len(assistant_spans) == 1 and normalized_messages[-1]["role"] == "assistant":
        terminal_start = supervised_indices[-1] + 1
        if terminal_start < len(input_ids_list):
            special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
            decode = getattr(tokenizer, "decode", None)
            terminal_control_indices: list[int] = []
            for index in range(terminal_start, len(input_ids_list)):
                token_id = int(input_ids_list[index])
                decoded = decode([token_id], skip_special_tokens=False) if callable(decode) else ""
                is_control = token_id in special_ids or (
                    isinstance(decoded, str) and decoded.startswith("<|") and decoded.endswith("|>")
                )
                if is_control:
                    terminal_control_indices.append(index)
            if terminal_control_indices:
                terminal_end = terminal_control_indices[-1]
                terminal_indices = list(range(terminal_start, terminal_end + 1))
                for index in terminal_indices:
                    assistant_mask_list[index] = True
                supervised_indices.extend(terminal_indices)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    assistant_mask = torch.tensor(assistant_mask_list, dtype=torch.bool)
    labels = input_ids.clone()
    labels.masked_fill_(~assistant_mask, IGNORE_INDEX)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    assistant_texts = [rendered_text[start:end] for start, end in assistant_spans]
    # The fixed workload has one assistant message, for which this is exactly
    # SHA-256(original assistant content).  Concatenation provides the natural
    # aggregate for multi-turn conversations; individual hashes are also kept.
    assistant_text = "".join(assistant_texts)
    metadata = TokenizationMetadata(
        sample_id=sample_id,
        total_token_count=total_token_count,
        supervised_token_count=len(supervised_indices),
        first_supervised_token_index=supervised_indices[0],
        last_supervised_token_index=supervised_indices[-1],
        rendered_text_sha256=_sha256_text(rendered_text),
        assistant_text_sha256=_sha256_text(assistant_text),
        tokenization_time_s=time.perf_counter() - started,
        assistant_char_spans=assistant_spans,
        assistant_text_sha256s=tuple(_sha256_text(text) for text in assistant_texts),
    )
    return TokenizedSample(
        sample_id=sample_id,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        assistant_mask=assistant_mask,
        offset_mapping=offsets,
        rendered_text=rendered_text,
        metadata=metadata,
    )


def load_jsonl_dataset(
    path: str | Path,
    tokenizer,
    *,
    max_sequence_length: int,
) -> list[TokenizedSample]:
    """Load a JSONL dataset strictly and preserve its physical line order."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"JSONL dataset not found: {dataset_path}")

    samples: list[TokenizedSample] = []
    seen_sample_ids: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{dataset_path}:{line_number}: blank JSONL row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{dataset_path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{dataset_path}:{line_number}: JSON row must be an object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise TypeError(
                    f"{dataset_path}:{line_number}: sample_id must be a non-empty string"
                )
            if sample_id in seen_sample_ids:
                raise ValueError(f"{dataset_path}:{line_number}: duplicate sample_id {sample_id!r}")
            if "messages" not in row:
                raise KeyError(f"{dataset_path}:{line_number}: missing messages")
            try:
                sample = tokenize_conversation(
                    tokenizer,
                    sample_id=sample_id,
                    messages=row["messages"],
                    max_sequence_length=max_sequence_length,
                )
            except Exception as exc:
                raise type(exc)(f"{dataset_path}:{line_number}: {exc}") from exc
            seen_sample_ids.add(sample_id)
            samples.append(sample)

    if not samples:
        raise ValueError(f"JSONL dataset is empty: {dataset_path}")
    return samples


def collate_tokenized_samples(
    samples: Iterable[TokenizedSample],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    """Pad samples without ever supervising padding positions."""

    items = list(samples)
    if not items:
        raise ValueError("at least one tokenized sample is required")
    max_length = max(item.input_ids.numel() for item in items)
    batch_size = len(items)
    input_ids = torch.full((batch_size, max_length), int(pad_token_id), dtype=torch.long)
    labels = torch.full((batch_size, max_length), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for row, item in enumerate(items):
        length = item.input_ids.numel()
        input_ids[row, :length].copy_(item.input_ids)
        labels[row, :length].copy_(item.labels)
        attention_mask[row, :length].copy_(item.attention_mask)
    return {
        "sample_ids": [item.sample_id for item in items],
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "metadata": [item.metadata for item in items],
    }


__all__ = [
    "IGNORE_INDEX",
    "TokenizationMetadata",
    "TokenizedSample",
    "collate_tokenized_samples",
    "load_jsonl_dataset",
    "load_real_tokenizer",
    "tokenize_conversation",
]
