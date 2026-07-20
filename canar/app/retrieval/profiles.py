from __future__ import annotations

from dataclasses import replace

from canar.app.retrieval.models import (
    DenseRetrievalParams,
    FusionRetrievalParams,
    ParentChildRetrievalParams,
    RerankRetrievalParams,
    RetrievalProfile,
    SparseRetrievalParams,
)

AGENT_RETRIEVAL_PROFILES: dict[str, str | None] = {
    "generic_agent": "hybrid_rerank_bge",
    "r_helpdesk": "simple_vector",
    "sas_to_r": None,
}

RERANKER_MODELS = {
    "bge": "bge-v2-m3",
    "qwen_0.6b": "qwen-0.6b",
    "qwen_4b": "qwen-4b",
    "qwen_8b": "qwen-8b",
}


def build_retrieval_profiles(
    collections: tuple[str, ...],
    dense_vector_name: str | None = None,
    sparse_vector_name: str | None = None,
) -> dict[str, RetrievalProfile]:
    dense = DenseRetrievalParams(vector_name=dense_vector_name or None)
    sparse = SparseRetrievalParams(vector_name=sparse_vector_name or None)
    parent_child = ParentChildRetrievalParams()

    simple_vector = RetrievalProfile(
        name="simple_vector",
        strategy="simple_vector",
        collections=collections,
        min_score=0.8,
        dense=dense,
    )
    simple_sparse = RetrievalProfile(
        name="simple_sparse",
        strategy="simple_sparse",
        collections=collections,
        min_score=0.8,
        sparse=sparse,
    )
    hybrid = RetrievalProfile(
        name="hybrid",
        strategy="hybrid",
        collections=collections,
        min_score=0.8,
        fallback_top_k=5,
        dense=dense,
        sparse=sparse,
        fusion=FusionRetrievalParams(),
    )

    # raise only min_score to 0.9
    # simple_vector = replace(simple_vector, min_score=0.9)
    # simple_sparse = replace(simple_sparse, min_score=0.9)
    # hybrid = replace(hybrid, min_score=0.9)

    profiles = {
        simple_vector.name: simple_vector,
        "simple_vector_parent_child": replace(
            simple_vector,
            name="simple_vector_parent_child",
            parent_child=parent_child,
        ),
        simple_sparse.name: simple_sparse,
        "simple_sparse_parent_child": replace(
            simple_sparse,
            name="simple_sparse_parent_child",
            parent_child=parent_child,
        ),
        hybrid.name: hybrid,
        "hybrid_parent_child": replace(
            hybrid,
            name="hybrid_parent_child",
            parent_child=parent_child,
        ),
    }

    default_rerank = RerankRetrievalParams(output_top_k=5)
    rerank_base = replace(
        hybrid,
        name="hybrid_rerank_bge",
        fetch_top_k=20,
        output_top_k=20,
        rerank=default_rerank,
    )
    for suffix, model in RERANKER_MODELS.items():
        rerank_profile = replace(
            rerank_base,
            name=f"hybrid_rerank_{suffix}",
            rerank=replace(default_rerank, model=model),
        )
        profiles[rerank_profile.name] = rerank_profile

        parent_rerank_profile = replace(
            rerank_profile,
            name=f"hybrid_parent_child_rerank_{suffix}",
            parent_child=parent_child,
        )
        profiles[parent_rerank_profile.name] = parent_rerank_profile

    return profiles
