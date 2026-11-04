"""Loader interface: a parser turns one dataset file into DatasetItem rows."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .item import DatasetItem


class DatasetLoader(Protocol):
    def __call__(self, path: Path) -> list[DatasetItem]: ...
