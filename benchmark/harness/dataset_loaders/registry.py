"""
Extension -> loader registry and the public ``load_dataset`` dispatcher.

A loader registers itself for one or more extensions with ``@register_loader``.
``load_dataset`` picks the loader from the file extension, or from an explicit
``fmt`` override ("declared format"). An unknown format raises
``UnsupportedDatasetFormat`` with a message listing what *is* supported.
"""

from __future__ import annotations

from pathlib import Path

from .base import DatasetLoader
from .item import DatasetItem

_LOADERS: dict[str, DatasetLoader] = {}


class UnsupportedDatasetFormat(ValueError):
    """Raised when no loader is registered for a dataset's format."""


def _normalise_ext(ext: str) -> str:
    ext = ext.lower()
    return ext if ext.startswith(".") else f".{ext}"


def register_loader(*extensions: str):
    """Decorator: register ``fn`` as the loader for the given extension(s)."""

    def decorator(fn: DatasetLoader) -> DatasetLoader:
        for ext in extensions:
            _LOADERS[_normalise_ext(ext)] = fn
        return fn

    return decorator


def supported_formats() -> list[str]:
    """The registered extensions, sorted (for help text and error messages)."""
    return sorted(_LOADERS)


def get_loader(ext: str) -> DatasetLoader:
    loader = _LOADERS.get(_normalise_ext(ext))
    if loader is None:
        supported = ", ".join(supported_formats()) or "(none registered)"
        raise UnsupportedDatasetFormat(
            f"Unsupported dataset format {_normalise_ext(ext)!r}. "
            f"Supported formats: {supported}."
        )
    return loader


def load_dataset(path: str | Path, fmt: str | None = None) -> list[DatasetItem]:
    """Load ``path`` into DatasetItem rows, dispatching by extension or ``fmt``."""
    path = Path(path)
    return get_loader(fmt or path.suffix)(path)
