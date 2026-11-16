import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import BenchmarkConfig


def _size(value: str) -> int | None:
    if value == "full":
        return None
    if value.isdecimal() and int(value) > 0:
        return int(value)
    raise argparse.ArgumentTypeError("stage must be a positive integer or 'full'")


def main() -> None:
    defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(description="Run the local MultiHop-RAG retrieval baselines")
    parser.add_argument("--sizes", type=_size, nargs="+", default=defaults.stage_sizes)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_directory)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--graph-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("stages must be unique")
    config = replace(defaults, output_directory=args.output_dir, stage_sizes=tuple(args.sizes))

    if args.audit_only:
        from .dataset import load_dataset

        data = load_dataset(config.raw_directory).audit
        print(json.dumps(data, ensure_ascii=True, indent=2))
    elif args.graph_only:
        from .graph_regressions import run_graph_regressions

        data = run_graph_regressions(config.example_corpus)
        print(json.dumps(data, ensure_ascii=True, indent=2))
    else:
        from .benchmark import run_benchmark

        run_benchmark(config)


if __name__ == "__main__":
    main()
