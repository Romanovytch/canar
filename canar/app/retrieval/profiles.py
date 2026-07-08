from __future__ import annotations

from canar.app.retrieval.models import (
    DenseRetrievalParams,
    FusionRetrievalParams,
    ParentChildRetrievalParams,
    RerankRetrievalParams,
    RetrievalProfile,
    SparseRetrievalParams,
)

AGENT_RETRIEVAL_PROFILES: dict[str, str | None] = {
    "r_helpdesk": "simple_vector",
    "sas_to_r": None,
}


def _dense(fetch_top_k: int) -> DenseRetrievalParams:
    return DenseRetrievalParams(
        fetch_top_k=fetch_top_k,
        min_score=0.35,
        max_kept=None,
    )


def _sparse(fetch_top_k: int) -> SparseRetrievalParams:
    return SparseRetrievalParams(
        fetch_top_k=fetch_top_k,
        min_score_ratio=0.35,
        gap_ratio=None,
        max_kept=None,
    )


def _fusion(output_top_k: int) -> FusionRetrievalParams:
    return FusionRetrievalParams(
        method="rrf",
        rrf_k=60,
        weights={"dense": 1.0, "sparse": 1.0},
        output_top_k=output_top_k,
    )


def _rerank(output_top_k: int) -> RerankRetrievalParams:
    return RerankRetrievalParams(
        output_top_k=output_top_k,
        reranker_name="bge",
        device="auto",
        max_length=8192,
    )


def build_retrieval_profiles(
    collections: tuple[str, ...],
    dense_vector_name: str | None = None,
    sparse_vector_name: str | None = None,
) -> dict[str, RetrievalProfile]:
    return {
        "simple_vector": RetrievalProfile(
            name="simple_vector",
            strategy="simple_vector",
            collections=collections,
            score_threshold=0.35,
            source_filter=None,
            fallback_top_k=3,
            vector_name=dense_vector_name or None,
            dense=_dense(5),
        ),
        "simple_vector_parent_child": RetrievalProfile(
            name="simple_vector_parent_child",
            strategy="simple_vector",
            collections=collections,
            score_threshold=0.35,
            source_filter="utilitr",
            fallback_top_k=3,
            vector_name=dense_vector_name or None,
            dense=_dense(5),
            parent_child=ParentChildRetrievalParams(
                parent_collection_suffix="_parent",
            ),
        ),
        "simple_sparse": RetrievalProfile(
            name="simple_sparse",
            strategy="simple_sparse",
            collections=collections,
            score_threshold=0.35,
            source_filter=None,
            fallback_top_k=3,
            vector_name=sparse_vector_name or None,
            sparse=_sparse(5),
        ),
        "simple_sparse_parent_child": RetrievalProfile(
            name="simple_sparse_parent_child",
            strategy="simple_sparse",
            collections=collections,
            score_threshold=0.35,
            source_filter=None,
            fallback_top_k=3,
            vector_name=sparse_vector_name or None,
            sparse=_sparse(5),
            parent_child=ParentChildRetrievalParams(
                parent_collection_suffix="_parent",
            ),
        ),
        "hybrid": RetrievalProfile(
            name="hybrid",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.35,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=_dense(10),
            sparse=_sparse(10),
            fusion=_fusion(5),
        ),
        "hybrid_rerank": RetrievalProfile(
            name="hybrid_rerank",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.35,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=_dense(20),
            sparse=_sparse(20),
            # On rerank profiles this is the RRF candidate pool sent to rerank.
            fusion=_fusion(20),
            rerank=_rerank(5),
        ),
        "hybrid_parent_child": RetrievalProfile(
            name="hybrid_parent_child",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.35,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=_dense(10),
            sparse=_sparse(10),
            fusion=_fusion(5),
            parent_child=ParentChildRetrievalParams(
                parent_collection_suffix="_parent",
            ),
        ),
        "hybrid_parent_child_rerank": RetrievalProfile(
            name="hybrid_parent_child_rerank",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.35,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=_dense(20),
            sparse=_sparse(20),
            # On rerank profiles this is the RRF candidate pool sent to rerank.
            fusion=_fusion(20),
            parent_child=ParentChildRetrievalParams(
                parent_collection_suffix="_parent",
            ),
            rerank=_rerank(5),
        ),
    }
