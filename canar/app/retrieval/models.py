from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RetrievalStrategyName = Literal["simple_vector", "simple_sparse", "hybrid"]
FusionMethod = Literal["rrf", "weighted_rrf"]


def _validate_retriever_overrides(
    retriever: str,
    fetch_top_k: int | None,
    min_score: float | None,
) -> None:
    if fetch_top_k is not None and fetch_top_k <= 0:
        raise ValueError(f"{retriever}.fetch_top_k must be greater than 0")
    if min_score is not None and not 0.0 <= min_score <= 1.0:
        raise ValueError(f"{retriever}.min_score must be between 0 and 1")


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class DenseRetrievalParams:
    vector_name: str | None = None
    fetch_top_k: int | None = None
    min_score: float | None = None

    def __post_init__(self) -> None:
        _validate_retriever_overrides("dense", self.fetch_top_k, self.min_score)


@dataclass(frozen=True)
class SparseRetrievalParams:
    vector_name: str | None = None
    fetch_top_k: int | None = None
    min_score: float | None = None
    gap_ratio: float | None = None

    def __post_init__(self) -> None:
        _validate_retriever_overrides("sparse", self.fetch_top_k, self.min_score)
        if self.gap_ratio is not None and not 0.0 <= self.gap_ratio <= 1.0:
            raise ValueError("sparse.gap_ratio must be between 0 and 1")


@dataclass(frozen=True)
class FusionRetrievalParams:
    method: FusionMethod = "rrf"
    rrf_k: int = 60
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "dense": 1.0,
            "sparse": 1.0,
        }
    )
    output_top_k: int | None = None

    def __post_init__(self) -> None:
        if self.rrf_k <= 0:
            raise ValueError("fusion.rrf_k must be greater than 0")
        if self.output_top_k is not None and self.output_top_k <= 0:
            raise ValueError("fusion.output_top_k must be greater than 0")


@dataclass(frozen=True)
class ParentChildRetrievalParams:
    parent_collection_suffix: str = "_parent"

    def __post_init__(self) -> None:
        if not self.parent_collection_suffix:
            raise ValueError("parent_child.parent_collection_suffix must not be empty")


@dataclass(frozen=True)
class SummaryRetrievalParams:
    collection_suffix: str


@dataclass(frozen=True)
class RerankRetrievalParams:
    output_top_k: int | None = None
    model: str = "bge-v2-m3"
    device: str | None = "auto"
    max_length: int = 8192

    def __post_init__(self) -> None:
        if self.output_top_k is not None and self.output_top_k <= 0:
            raise ValueError("rerank.output_top_k must be greater than 0")
        if not self.model:
            raise ValueError("rerank.model must not be empty")
        if self.max_length <= 0:
            raise ValueError("rerank.max_length must be greater than 0")


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    strategy: RetrievalStrategyName
    collections: tuple[str, ...]
    fetch_top_k: int = 10
    min_score: float = 0.75
    fallback_top_k: int = 3
    output_top_k: int = 5
    source_filter: str | None = None
    dense: DenseRetrievalParams | None = None
    sparse: SparseRetrievalParams | None = None
    fusion: FusionRetrievalParams | None = None
    rerank: RerankRetrievalParams | None = None
    parent_child: ParentChildRetrievalParams | None = None
    summary: SummaryRetrievalParams | None = None

    def effective_collections(self) -> tuple[str, ...]:
        if self.summary is None:
            return self.collections
        suffix = self.summary.collection_suffix.strip()
        return tuple(f"{collection}{suffix}" for collection in self.collections)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile.name must not be empty")
        if not self.collections:
            raise ValueError("profile.collections must not be empty")
        if self.fetch_top_k <= 0:
            raise ValueError("profile.fetch_top_k must be greater than 0")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("profile.min_score must be between 0 and 1")
        if self.fallback_top_k < 0:
            raise ValueError("profile.fallback_top_k must be greater than or equal to 0")
        if self.output_top_k <= 0:
            raise ValueError("profile.output_top_k must be greater than 0")

        required_blocks = {
            "simple_vector": (("dense", self.dense),),
            "simple_sparse": (("sparse", self.sparse),),
            "hybrid": (
                ("dense", self.dense),
                ("sparse", self.sparse),
                ("fusion", self.fusion),
            ),
        }
        if self.strategy not in required_blocks:
            raise ValueError(f"unsupported retrieval strategy: {self.strategy!r}")
        missing = [name for name, value in required_blocks[self.strategy] if value is None]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{self.strategy} profile requires parameter block(s): {joined}")
        if self.rerank is not None and self.strategy != "hybrid":
            raise ValueError("rerank is only supported by hybrid profiles")

        if self.summary is None:
            return
        if self.strategy != "hybrid":
            raise ValueError("summary retrieval requires strategy='hybrid'")
        if not self.summary.collection_suffix.strip():
            raise ValueError("summary retrieval requires a non-empty collection suffix")
        if self.parent_child is not None:
            raise ValueError("summary retrieval cannot be combined with parent retrieval")


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    profile_name: str
    dense_vector: list[float] | None = None
    sparse_vector: SparseVector | None = None


@dataclass(frozen=True)
class RetrievalHit:
    text: str
    collection: str
    score: float
    score_norm: float
    generation_text: str | None = None
    source: str | None = None
    source_url: str | None = None
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None
