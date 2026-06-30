from __future__ import annotations

from canar.app.retrieval.models import (
    DenseRetrievalParams,
    FusionRetrievalParams,
    ParentChildRetrievalParams,
    SparseRetrievalParams,
)
from canar.app.retrieval.profiles import AGENT_RETRIEVAL_PROFILES, build_retrieval_profiles


def test_expanded_vector_profiles_are_registered_without_changing_agent_mapping():
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
    assert parent_child.dense == DenseRetrievalParams(
        top_k=5,
        min_score=0.35,
        max_kept=None,
    )
    assert parent_child.parent_child == ParentChildRetrievalParams(
        parent_collection_suffix="_parent",
    )

    assert parent_child_hybrid.strategy == "parent_child_hybrid"
    assert parent_child_hybrid.collections == ("children_a", "children_b")
    assert parent_child_hybrid.vector_name == "text-sparse"
    assert parent_child_hybrid.dense == DenseRetrievalParams(
        top_k=10,
        min_score=0.35,
        max_kept=None,
    )
    assert parent_child_hybrid.sparse == SparseRetrievalParams(
        top_k=10,
        min_score_ratio=0.35,
        gap_ratio=None,
        max_kept=None,
    )
    assert parent_child_hybrid.fusion == FusionRetrievalParams(
        method="rrf",
        rrf_k=60,
        weights={"dense": 1.0, "sparse": 1.0},
        final_top_k=5,
    )
    assert parent_child_hybrid.parent_child == ParentChildRetrievalParams(
        parent_collection_suffix="_parent",
    )
    assert AGENT_RETRIEVAL_PROFILES["r_helpdesk"] == "simple_vector"
