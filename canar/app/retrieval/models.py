from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class DenseRetrievalParams:
    fetch_top_k: int = 5
    min_score: float | None = None
    max_kept: int | None = None


@dataclass(frozen=True)
class SparseRetrievalParams:
    fetch_top_k: int = 5
    min_score_ratio: float | None = None
    gap_ratio: float | None = None
    max_kept: int | None = None


@dataclass(frozen=True)
class FusionRetrievalParams:
    method: str = "rrf"
    rrf_k: int = 60
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "dense": 1.0,
            "sparse": 1.0,
        }
    )
    output_top_k: int = 5


@dataclass(frozen=True)
class ParentChildRetrievalParams:
    parent_collection_suffix: str = "_parent"


@dataclass(frozen=True)
class RerankRetrievalParams:
    output_top_k: int = 5
    # Profiles choose the reranker family; exact model overrides stay below this layer.
    reranker_name: str = "bge"
    device: str | None = "auto"
    max_length: int = 8192


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    strategy: str
    collections: tuple[str, ...]
    score_threshold: float = 0.35
    source_filter: str | None = "utilitr"
    fallback_top_k: int = 3
    vector_name: str | None = None
    dense: DenseRetrievalParams | None = None
    sparse: SparseRetrievalParams | None = None
    fusion: FusionRetrievalParams | str | None = None
    rerank: RerankRetrievalParams | None = None
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    parent_child: ParentChildRetrievalParams | None = None
    parent_collection_suffix: str = "_parent"

    def dense_params(self) -> DenseRetrievalParams:
        if self.dense is not None:
            return self.dense
        return DenseRetrievalParams(
            fetch_top_k=5,
            min_score=self.score_threshold,
        )

    def sparse_params(self) -> SparseRetrievalParams:
        if self.sparse is not None:
            return self.sparse
        return SparseRetrievalParams(fetch_top_k=5)

    def fusion_params(self) -> FusionRetrievalParams:
        if isinstance(self.fusion, FusionRetrievalParams):
            return self.fusion
        return FusionRetrievalParams(
            method=self.fusion or "rrf",
            rrf_k=self.rrf_k,
            weights={
                "dense": self.dense_weight,
                "sparse": self.sparse_weight,
            },
            output_top_k=5,
        )

    def parent_child_params(self) -> ParentChildRetrievalParams:
        if self.parent_child is not None:
            return self.parent_child
        return ParentChildRetrievalParams(
            parent_collection_suffix=self.parent_collection_suffix,
        )

    def rerank_params(self) -> RerankRetrievalParams | None:
        if self.rerank is not None:
            return self.rerank
        return None


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
