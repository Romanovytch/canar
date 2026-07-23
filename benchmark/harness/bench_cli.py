"""Command-line parsing for the end-to-end benchmark entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_eval_args(
    argv: list[str] | None = None,
    *,
    default_config: Path,
) -> argparse.Namespace:
    """Parse and validate benchmark-only command-line options."""
    parser = argparse.ArgumentParser(
        description="Run the end-to-end CanaR retrieval and RAGAS benchmark."
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=None,
        help=(
            "Benchmark YAML file. Can be passed multiple times; relative paths "
            "are resolved from the current directory."
        ),
    )
    parser.add_argument(
        "--retrieval-k",
        "--retrieval_k",
        type=int,
        default=None,
        help="Cutoff for retrieval metrics; default: score every returned result.",
    )
    args = parser.parse_args(argv)
    if args.retrieval_k is not None and args.retrieval_k <= 0:
        parser.error("--retrieval-k must be greater than zero")

    configs = args.config or [default_config]
    resolved_configs = []
    for config in configs:
        config = config.expanduser()
        if not config.is_absolute():
            config = Path.cwd() / config
        resolved_configs.append(config.resolve())
    args.configs = resolved_configs
    # Backward-compatible convenience for callers that still expect one config.
    args.config = resolved_configs[0]
    return args
