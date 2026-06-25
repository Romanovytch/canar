from __future__ import annotations

from canar.app.retrieval.models import RetrievalProfile

AGENT_RETRIEVAL_PROFILES: dict[str, str | None] = {
    "r_helpdesk": "simple_vector",
    "sas_to_r": None,
}


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
            top_k=5,
            score_threshold=0.35,
            source_filter="utilitr",
            fallback_top_k=3,
            vector_name=dense_vector_name or None,
        ),
        "simple_sparse": RetrievalProfile(
            name="simple_sparse",
            strategy="simple_sparse",
            collections=collections,
            top_k=5,
            score_threshold=0.35,
            source_filter="utilitr",
            fallback_top_k=3,
            vector_name=sparse_vector_name or None,
        ),
        "hybrid": RetrievalProfile(
            name="hybrid",
            strategy="hybrid",
            collections=collections,
            top_k=5,
            score_threshold=0.35,
            source_filter="utilitr",
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense_top_k=10,
            sparse_top_k=10,
            fusion="rrf",
            final_top_k=5,
            rrf_k=10,
            dense_weight=1.0,
            sparse_weight=1.0,
        ),
    }
