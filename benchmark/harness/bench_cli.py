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
        default=default_config,
        help="Benchmark YAML file (relative paths are resolved from the current directory).",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=None,
        help="Cutoff for retrieval metrics; default: score every returned result.",
    )
    args = parser.parse_args(argv)
    if args.retrieval_k is not None and args.retrieval_k <= 0:
        parser.error("--retrieval-k must be greater than zero")

    config = args.config.expanduser()
    if not config.is_absolute():
        config = Path.cwd() / config
    args.config = config.resolve()
    return args
