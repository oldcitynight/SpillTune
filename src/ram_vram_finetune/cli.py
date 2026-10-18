import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(prog="spilltune", description="Independent synchronous block streaming LoRA")
    from . import __version__
    parser.add_argument("--version", action="version", version=f"spilltune {__version__} (Fire up)")
    sub = parser.add_subparsers(dest="command", required=True)
    tiny = sub.add_parser("make-tiny", help="generate random tiny native HF fixture; no download")
    tiny.add_argument("--output", required=True)
    tiny.add_argument("--layers", type=int, default=4)
    tiny.add_argument('--adapter', choices=['qwen35', 'llama', 'qwen2','qwen3','qwen3_moe', 'deepseek_v3','gpt_oss','glm4_moe_lite','qwen35_moe'], default='qwen35')
    inspect = sub.add_parser("inspect", help="read model metadata without loading tensor payloads")
    inspect.add_argument("--model", required=True)
    inspect.add_argument('--adapter', choices=['qwen35', 'llama', 'qwen2','qwen3','qwen3_moe', 'deepseek_v3','gpt_oss','glm4_moe_lite','qwen35_moe'], default='qwen35')
    quantize=sub.add_parser('quantize',help='export internal group-128 compressed base, not GGUF')
    quantize.add_argument('--model',required=True);quantize.add_argument('--output',required=True)
    quantize.add_argument('--adapter',choices=['qwen35','llama','qwen2','deepseek_v3'],required=True)
    from .quantization.codecs import FORMATS
    quantize.add_argument('--format',choices=FORMATS,required=True)
    for name in ("train", "resume"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "resume":
            command.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    try:
        if args.command == "make-tiny":
            from .fixture import make_tiny
            print(make_tiny(args.output, layers=args.layers, adapter=args.adapter))
        elif args.command=='quantize':
            from .quantization.storage import export_quantized
            print(json.dumps(export_quantized(args.model,args.output,args.adapter,args.format),indent=2))
        elif args.command == "inspect":
            from .adapters.registry import inspect_model
            value=inspect_model(args.model,args.adapter)
            value['eligible_quantization_matrices']=len(value['quantization_matrix_keys'])
            for key in ("entries", "required", "text_config", 'config','runtime_shapes','quantization_matrix_keys','quantization_fp32_keys'):
                value.pop(key, None)
            print(json.dumps(value, indent=2))
        else:
            from pathlib import Path
            from .config import TrainConfig
            from .training.engine import run
            config = TrainConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
            print(json.dumps(run(config, resume=getattr(args, "checkpoint", None)), indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, TypeError, KeyError) as exc:
        print(f"spilltune: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
