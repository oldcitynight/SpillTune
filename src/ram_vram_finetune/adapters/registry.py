"""Explicit adapter registry. Configuration never imports arbitrary remote code."""
_FACTORIES = {}
_INSPECTORS = {}


def register(name, factory, *, inspector=None):
    if not isinstance(name, str) or not name or name in _FACTORIES or name in {'qwen35', 'llama', 'qwen2','qwen3','qwen3_moe','deepseek_v3','gpt_oss','glm4_moe_lite','qwen35_moe'}:
        raise ValueError("invalid or already registered adapter")
    if not callable(factory):
        raise TypeError("adapter factory must be callable")
    if inspector is not None and not callable(inspector):raise TypeError('inspector must be callable')
    _FACTORIES[name] = factory
    if inspector is not None:
        _INSPECTORS[name]=inspector


def resolve(name):
    if name in _FACTORIES:
        return _FACTORIES[name]
    if name == "qwen35_moe":
        from .qwen35_moe import Qwen35MoeAdapter
        return Qwen35MoeAdapter
    if name == "qwen35":
        from .qwen35.adapter import Qwen35Adapter
        return Qwen35Adapter
    if name in {'llama', 'qwen2','qwen3'}:
        from .dense import LlamaAdapter, Qwen2Adapter, Qwen3Adapter
        return {'llama': LlamaAdapter, 'qwen2': Qwen2Adapter,'qwen3':Qwen3Adapter}[name]
    if name=='qwen3_moe':
        from .qwen3_moe import Qwen3MoeAdapter
        return Qwen3MoeAdapter
    if name=='glm4_moe_lite':
        from .glm4_moe_lite import Glm4MoeLiteAdapter
        return Glm4MoeLiteAdapter
    if name=='gpt_oss':
        from .gpt_oss.adapter import GptOssAdapter
        return GptOssAdapter
    if name=='deepseek_v3':
        from .deepseek import DeepseekV3Adapter
        return DeepseekV3Adapter
    raise ValueError(f"unsupported_architecture: no registered adapter {name!r}")


def inspect_model(path,name):
    if name in _INSPECTORS:return _INSPECTORS[name](path)
    if name=='qwen35_moe':
        from .qwen35_moe import inspect_qwen35_moe
        return inspect_qwen35_moe(path)
    if name=='qwen35':
        from .qwen35.adapter import inspect_model as inspect_qwen
        return inspect_qwen(path)
    if name in {'llama','qwen2','qwen3'}:
        from .dense import inspect_dense
        return inspect_dense(path,name)
    if name=='qwen3_moe':
        from .qwen3_moe import inspect_qwen3_moe
        return inspect_qwen3_moe(path)
    if name=='glm4_moe_lite':
        from .glm4_moe_lite import inspect_glm4_moe_lite
        return inspect_glm4_moe_lite(path)
    if name=='gpt_oss':
        from .gpt_oss.adapter import inspect_gpt_oss
        return inspect_gpt_oss(path)
    if name=='deepseek_v3':
        from .deepseek import inspect_deepseek
        return inspect_deepseek(path)
    raise ValueError(f'unsupported_architecture: no metadata inspector for {name!r}')
