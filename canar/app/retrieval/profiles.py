from __future__ import annotations

from canar.app.retrieval.models import RetrievalProfile

AGENT_RETRIEVAL_PROFILES: dict[str, str | None] = {
    "r_helpdesk": "simple_vector",
    "sas_to_r": None,
}


def build_retrieval_profiles(
    collections: tuple[str, ...],
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
    }
