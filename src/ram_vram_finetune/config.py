import dataclasses
import math
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    model: str
    dataset: str
    output: str
    adapter: str = "qwen35"
    device: str = "cpu"
    steps: int = 2
    accumulation: int = 1
    rank: int = 2
    alpha: float = 4.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    token_chunk: int = 8
    max_sequence_length: int = 256
    host_budget_bytes: int = 1024**3
    device_budget_bytes: int = 2 * 1024**3
    checkpoint_every: int = 1
    quantization: str = 'none'
    quantized_compute: str = 'bf16'
    transfer_mode: str = 'synchronous'
    optimizer_validation: str = 'scalar'
    prefetch_pool_budget_bytes: int = 128*1024**2
    prefetch_max_slots: int = 2
    prefetch_reserve_bytes: int = 256*1024**2

    def __post_init__(self):
        if self.transfer_mode not in ('synchronous', 'batched', 'prefetch'):
            raise ValueError('unsupported_capability: transfer_mode must be synchronous, batched or prefetch')
        if self.optimizer_validation not in ('scalar', 'batched'):
            raise ValueError('unsupported_capability: optimizer_validation must be scalar or batched')
        if self.transfer_mode in ('batched','prefetch') and (self.device != 'cuda:0' or self.quantization != 'none'):
            raise ValueError('unsupported_capability: batched/prefetch transfer requires CUDA and uncompressed weights')
        for key in ("steps", "accumulation", "rank", "token_chunk", "max_sequence_length",
                    "host_budget_bytes", "device_budget_bytes", "checkpoint_every", "prefetch_pool_budget_bytes",
                    "prefetch_max_slots", "prefetch_reserve_bytes"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for key in ("alpha", "learning_rate", "weight_decay"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.device != "cpu" and self.device != "cuda:0":
            raise ValueError("unsupported_capability: device must be cpu or cuda:0")
        if not isinstance(self.adapter, str) or not self.adapter:
            raise ValueError("adapter must be an explicit registered name")
        if self.quantization!='none':
            from .quantization.codecs import check_format
            check_format(self.quantization)
        if self.quantized_compute!='bf16':raise ValueError('unsupported_quantization: native low precision compute is not implemented')
        for key in ("model", "dataset", "output"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be an explicit local path")

    def semantics(self):
        value = dataclasses.asdict(self)
        # Execution policy does not change mathematical/checkpoint identity.
        value.pop('transfer_mode'); value.pop('optimizer_validation')
        for key in ('prefetch_pool_budget_bytes','prefetch_max_slots','prefetch_reserve_bytes'):value.pop(key)
        if self.quantization=='none':
            value.pop('quantization');value.pop('quantized_compute')
        for key in ("output", "steps", "checkpoint_every"):
            value.pop(key)
        for key in ("model", "dataset"):
            value[key] = str(Path(value[key]).resolve())
        if self.quantization!='none':value['model']='content-addressed-quantized-base'
        return value
