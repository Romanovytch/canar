from __future__ import annotations

from canar.app.retrieval.models import (
    DenseRetrievalParams,
    FusionRetrievalParams,
    ParentChildRetrievalParams,
    RerankRetrievalParams,
    SparseRetrievalParams,
)
from canar.app.retrieval.profiles import AGENT_RETRIEVAL_PROFILES, build_retrieval_profiles


def test_expanded_profiles_are_registered_without_changing_agent_mapping():
    profiles = build_retrieval_profiles(
        ("children_a", "children_b"),
        dense_vector_name="text-dense",
        sparse_vector_name="text-sparse",
    )

    vector_parent_child = profiles["simple_vector_parent_child"]
    sparse_parent_child = profiles["simple_sparse_parent_child"]
    hybrid_parent_child = profiles["hybrid_parent_child"]

    assert vector_parent_child.strategy == "simple_vector"
    assert vector_parent_child.collections == ("children_a", "children_b")
    assert vector_parent_child.vector_name == "text-dense"
    assert vector_parent_child.dense == DenseRetrievalParams(
        top_k=5,
        min_score=0.35,
        max_kept=None,
    )
    assert vector_parent_child.parent_child == ParentChildRetrievalParams(
        parent_collection_suffix="_parent",
    )

    assert sparse_parent_child.strategy == "simple_sparse"
    assert sparse_parent_child.vector_name == "text-sparse"
    assert sparse_parent_child.sparse == SparseRetrievalParams(
        top_k=5,
        min_score_ratio=0.35,
        gap_ratio=None,
        max_kept=None,
    )
    assert sparse_parent_child.parent_child == ParentChildRetrievalParams(
        parent_collection_suffix="_parent",
    )

    assert hybrid_parent_child.strategy == "hybrid"
    assert hybrid_parent_child.collections == ("children_a", "children_b")
    assert hybrid_parent_child.vector_name == "text-sparse"
    assert hybrid_parent_child.dense == DenseRetrievalParams(
        top_k=10,
        min_score=0.35,
        max_kept=None,
    )
    assert hybrid_parent_child.sparse == SparseRetrievalParams(
        top_k=10,
        min_score_ratio=0.35,
        gap_ratio=None,
        max_kept=None,
    )
    assert hybrid_parent_child.fusion == FusionRetrievalParams(
        method="rrf",
        rrf_k=60,
        weights={"dense": 1.0, "sparse": 1.0},
        final_top_k=5,
    )
    assert hybrid_parent_child.rerank == RerankRetrievalParams(
        candidate_top_k=20,
        rerank_top_n=5,
        reranker_model=None,
    )
    assert hybrid_parent_child.parent_child == ParentChildRetrievalParams(
        parent_collection_suffix="_parent",
    )
    assert AGENT_RETRIEVAL_PROFILES["r_helpdesk"] == "simple_vector"


def test_expanded_vector_profiles_are_registered_with_hybrid_helpdesk_default():
    profiles = build_retrieval_profiles(
        ("children_a", "children_b"),
        dense_vector_name="text-dense",
        sparse_vector_name="text-sparse",
    )

    parent_child = profiles["parent_child_vector"]
    parent_child_hybrid = profiles["parent_child_hybrid"]

    assert parent_child.strategy == "parent_child_vector"
    assert parent_child.collections == ("children_a", "children_b")
    assert parent_child.vector_name == "text-dense"
    assert parent_child_hybrid.strategy == "parent_child_hybrid"
    assert parent_child_hybrid.collections == ("children_a", "children_b")
    assert parent_child_hybrid.vector_name == "text-sparse"
    assert AGENT_RETRIEVAL_PROFILES["r_helpdesk"] == "hybrid"
