"""
Format-aware benchmark dataset loading.

The benchmark used to read its questions with a single ``pd.read_csv`` in
``ragas_bench.run_benchmark``. This package replaces that with a small
abstraction so a dataset file is turned into a list of project-owned
``DatasetItem`` objects, with the parser chosen by file extension (or an
explicit ``fmt`` override). Adding a new format (e.g. JSON) is a new loader
module that registers itself — no change to the benchmark script.

Modelled on AgoRa's ``agora/sources`` loader/registry pattern.

    load_dataset(path)            -> list[DatasetItem]   (extension dispatch)
    register_loader(".csv")(fn)   -> register a parser for one or more extensions
    UnsupportedDatasetFormat                              raised on unknown formats
"""

from __future__ import annotations

from .item import DatasetItem
from .registry import (
    UnsupportedDatasetFormat,
    get_loader,
    load_dataset,
    register_loader,
    supported_formats,
)

__all__ = [
    "DatasetItem",
    "UnsupportedDatasetFormat",
    "get_loader",
    "load_dataset",
    "register_loader",
    "supported_formats",
]
