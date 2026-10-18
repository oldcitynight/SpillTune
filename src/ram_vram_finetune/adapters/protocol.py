"""Model-neutral plan: no architecture names, target names or layer-count gates."""
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from ..finetune.lora import LoRAProjection
from ..buffers import RawBufferContract


@dataclass
class BlockPlan:
    index: int
    kind: str
    functional: Callable
    kwargs: dict[str, Any]
    host_weights: dict[str, torch.Tensor]
    adapters: dict[str, LoRAProjection]
    storage_specs: dict | None = None
    buffer_contract: RawBufferContract | None = None


class ModelAdapter(Protocol):
    blocks: list[BlockPlan]
    embedding: torch.Tensor
    head: torch.Tensor
    identity: str
    device: torch.device

    def positions_and_masks(self, valid: torch.Tensor) -> tuple[tuple, dict]: ...
    def terminal_norm(self, hidden: torch.Tensor) -> torch.Tensor: ...
    def artifact_key(self, layer: int, target: str, parameter: str) -> str: ...
    def runtime_metadata(self) -> dict: ...
    def close(self) -> None: ...
