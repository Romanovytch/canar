from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    strategy: str
    collections: tuple[str, ...]
    top_k: int = 5
    score_threshold: float = 0.35
    source_filter: str | None = "utilitr"
    fallback_top_k: int = 3
    vector_name: str | None = None


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    profile_name: str
    dense_vector: list[float] | None = None


@dataclass(frozen=True)
class RetrievalHit:
    text: str
    collection: str
    score: float
    score_norm: float
    source: str | None = None
    source_url: str | None = None
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
