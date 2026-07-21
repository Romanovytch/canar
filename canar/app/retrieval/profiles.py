from __future__ import annotations

from canar.app.retrieval.models import (
    DenseRetrievalParams,
    FusionRetrievalParams,
    ParentChildRetrievalParams,
    RerankRetrievalParams,
    RetrievalProfile,
    SparseRetrievalParams,
    SummaryRetrievalParams,
)

AGENT_RETRIEVAL_PROFILES: dict[str, str | None] = {
    "generic_agent": "hybrid_summary",
    "r_helpdesk": "simple_vector",
    "sas_to_r": None,
}


def build_retrieval_profiles(
    collections: tuple[str, ...],
    dense_vector_name: str | None = None,
    sparse_vector_name: str | None = None,
) -> dict[str, RetrievalProfile]:
    profiles = {
        "simple_vector": RetrievalProfile(
            name="simple_vector",
            strategy="simple_vector",
            collections=collections,
            dense=DenseRetrievalParams(
                fetch_top_k=5,
                min_score=0.8,
                max_kept=None,
            ),
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=3,
            vector_name=dense_vector_name or None,
        ),
        "simple_vector_parent_child": RetrievalProfile(
            name="simple_vector_parent_child",
            strategy="simple_vector",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=3,
            vector_name=dense_vector_name or None,
            dense=DenseRetrievalParams(
                fetch_top_k=5,
                min_score=0.8,
                max_kept=None,
            ),
            parent_child=ParentChildRetrievalParams(
                parent_collection_suffix="_parent",
            ),
        ),
        "simple_sparse": RetrievalProfile(
            name="simple_sparse",
            strategy="simple_sparse",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=3,
            vector_name=sparse_vector_name or None,
            sparse=SparseRetrievalParams(
                fetch_top_k=5,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
        ),
        "simple_sparse_parent_child": RetrievalProfile(
            name="simple_sparse_parent_child",
            strategy="simple_sparse",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=3,
            vector_name=sparse_vector_name or None,
            sparse=SparseRetrievalParams(
                fetch_top_k=5,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
            parent_child=ParentChildRetrievalParams(
                parent_collection_suffix="_parent",
            ),
        ),
        "hybrid": RetrievalProfile(
            name="hybrid",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=DenseRetrievalParams(
                fetch_top_k=10,
                min_score=0.8,
                max_kept=None,
            ),
            sparse=SparseRetrievalParams(
                fetch_top_k=10,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
            fusion=FusionRetrievalParams(
                method="rrf",
                rrf_k=60,
                weights={"dense": 1.0, "sparse": 1.0},
                output_top_k=5,
            ),
        ),
        "hybrid_rerank_bge": RetrievalProfile(
            name="hybrid_rerank_bge",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=DenseRetrievalParams(
                fetch_top_k=20,
                min_score=0.8,
                max_kept=None,
            ),
            sparse=SparseRetrievalParams(
                fetch_top_k=20,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
            # On rerank profiles this is the RRF candidate pool sent to rerank.
            fusion=FusionRetrievalParams(
                method="rrf",
                rrf_k=60,
                weights={"dense": 1.0, "sparse": 1.0},
                output_top_k=20,
            ),
            rerank=RerankRetrievalParams(
                output_top_k=5,
                reranker_name="bge-v2-m3",
                device="auto",
                max_length=8192,
            ),
        ),
        "hybrid_rerank_qwen_0.6b": RetrievalProfile(
            name="hybrid_rerank_qwen_0.6b",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=DenseRetrievalParams(
                fetch_top_k=20,
                min_score=0.8,
                max_kept=None,
            ),
            sparse=SparseRetrievalParams(
                fetch_top_k=20,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
            # On rerank profiles this is the RRF candidate pool sent to rerank.
            fusion=FusionRetrievalParams(
                method="rrf",
                rrf_k=60,
                weights={"dense": 1.0, "sparse": 1.0},
                output_top_k=20,
            ),
            rerank=RerankRetrievalParams(
                output_top_k=5,
                reranker_name="qwen-0.6b",
                device="auto",
                max_length=8192,
            ),
        ),
        "hybrid_rerank_qwen_4b": RetrievalProfile(
            name="hybrid_rerank_qwen_4b",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=DenseRetrievalParams(
                fetch_top_k=20,
                min_score=0.8,
                max_kept=None,
            ),
            sparse=SparseRetrievalParams(
                fetch_top_k=20,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
            # On rerank profiles this is the RRF candidate pool sent to rerank.
            fusion=FusionRetrievalParams(
                method="rrf",
                rrf_k=60,
                weights={"dense": 1.0, "sparse": 1.0},
                output_top_k=20,
            ),
            rerank=RerankRetrievalParams(
                output_top_k=5,
                reranker_name="qwen-4b",
                device="auto",
                max_length=8192,
            ),
        ),
        "hybrid_rerank_qwen_8b": RetrievalProfile(
            name="hybrid_rerank_qwen_8b",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=DenseRetrievalParams(
                fetch_top_k=20,
                min_score=0.8,
                max_kept=None,
            ),
            sparse=SparseRetrievalParams(
                fetch_top_k=20,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
            # On rerank profiles this is the RRF candidate pool sent to rerank.
            fusion=FusionRetrievalParams(
                method="rrf",
                rrf_k=60,
                weights={"dense": 1.0, "sparse": 1.0},
                output_top_k=20,
            ),
            rerank=RerankRetrievalParams(
                output_top_k=5,
                reranker_name="qwen-8b",
                device="auto",
                max_length=8192,
            ),
        ),
        "hybrid_parent_child": RetrievalProfile(
            name="hybrid_parent_child",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=DenseRetrievalParams(
                fetch_top_k=10,
                min_score=0.8,
                max_kept=None,
            ),
            sparse=SparseRetrievalParams(
                fetch_top_k=10,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
            fusion=FusionRetrievalParams(
                method="rrf",
                rrf_k=60,
                weights={"dense": 1.0, "sparse": 1.0},
                output_top_k=5,
            ),
            parent_child=ParentChildRetrievalParams(
                parent_collection_suffix="_parent",
            ),
        ),
        "hybrid_parent_child_rerank_bge": RetrievalProfile(
            name="hybrid_parent_child_rerank_bge",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=DenseRetrievalParams(
                fetch_top_k=20,
                min_score=0.8,
                max_kept=None,
            ),
            sparse=SparseRetrievalParams(
                fetch_top_k=20,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
            # On rerank profiles this is the RRF candidate pool sent to rerank.
            fusion=FusionRetrievalParams(
                method="rrf",
                rrf_k=60,
                weights={"dense": 1.0, "sparse": 1.0},
                output_top_k=20,
            ),
            parent_child=ParentChildRetrievalParams(
                parent_collection_suffix="_parent",
            ),
            rerank=RerankRetrievalParams(
                output_top_k=5,
                reranker_name="bge-v2-m3",
                device="auto",
                max_length=8192,
            ),
        ),
        "hybrid_parent_child_rerank_qwen_0.6b": RetrievalProfile(
            name="hybrid_parent_child_rerank_qwen_0.6b",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=DenseRetrievalParams(
                fetch_top_k=20,
                min_score=0.8,
                max_kept=None,
            ),
            sparse=SparseRetrievalParams(
                fetch_top_k=20,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
            # On rerank profiles this is the RRF candidate pool sent to rerank.
            fusion=FusionRetrievalParams(
                method="rrf",
                rrf_k=60,
                weights={"dense": 1.0, "sparse": 1.0},
                output_top_k=20,
            ),
            parent_child=ParentChildRetrievalParams(
                parent_collection_suffix="_parent",
            ),
            rerank=RerankRetrievalParams(
                output_top_k=5,
                reranker_name="qwen-0.6b",
                device="auto",
                max_length=8192,
            ),
        ),
        "hybrid_parent_child_rerank_qwen_4b": RetrievalProfile(
            name="hybrid_parent_child_rerank_qwen_4b",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=DenseRetrievalParams(
                fetch_top_k=20,
                min_score=0.8,
                max_kept=None,
            ),
            sparse=SparseRetrievalParams(
                fetch_top_k=20,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
            # On rerank profiles this is the RRF candidate pool sent to rerank.
            fusion=FusionRetrievalParams(
                method="rrf",
                rrf_k=60,
                weights={"dense": 1.0, "sparse": 1.0},
                output_top_k=20,
            ),
            parent_child=ParentChildRetrievalParams(
                parent_collection_suffix="_parent",
            ),
            rerank=RerankRetrievalParams(
                output_top_k=5,
                reranker_name="qwen-4b",
                device="auto",
                max_length=8192,
            ),
        ),
        "hybrid_parent_child_rerank_qwen_8b": RetrievalProfile(
            name="hybrid_parent_child_rerank_qwen_8b",
            strategy="hybrid",
            collections=collections,
            score_threshold=0.8,
            source_filter=None,
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=DenseRetrievalParams(
                fetch_top_k=20,
                min_score=0.8,
                max_kept=None,
            ),
            sparse=SparseRetrievalParams(
                fetch_top_k=20,
                min_score_ratio=0.8,
                gap_ratio=None,
                max_kept=None,
            ),
            # On rerank profiles this is the RRF candidate pool sent to rerank.
            fusion=FusionRetrievalParams(
                method="rrf",
                rrf_k=60,
                weights={"dense": 1.0, "sparse": 1.0},
                output_top_k=20,
            ),
            parent_child=ParentChildRetrievalParams(
                parent_collection_suffix="_parent",
            ),
            rerank=RerankRetrievalParams(
                output_top_k=5,
                reranker_name="qwen-8b",
                device="auto",
                max_length=8192,
            ),
        ),
    }
    profiles["hybrid_summary"] = RetrievalProfile(
        name="hybrid_summary",
        strategy="hybrid",
        collections=collections,
        score_threshold=0.7,
        source_filter=None,
        fallback_top_k=5,
        vector_name=sparse_vector_name or None,
        dense=DenseRetrievalParams(
            fetch_top_k=10,
            min_score=0.7,
            max_kept=None,
        ),
        sparse=SparseRetrievalParams(
            fetch_top_k=10,
            min_score_ratio=0.7,
            gap_ratio=None,
            max_kept=None,
        ),
        fusion=FusionRetrievalParams(
            method="rrf",
            rrf_k=60,
            weights={"dense": 3.0, "sparse": 1.0 },
            output_top_k=2,
        ),
        summary=SummaryRetrievalParams(
            collection_suffix="_summaries",
        ),
    )
    return profiles
