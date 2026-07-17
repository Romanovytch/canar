from __future__ import annotations

from dataclasses import replace

from canar.app.retrieval.models import (
    DenseRetrievalParams,
    FusionRetrievalParams,
    ParentChildRetrievalParams,
    RerankRetrievalParams,
    SparseRetrievalParams,
)
from canar.app.retrieval.profiles import AGENT_RETRIEVAL_PROFILES, build_retrieval_profiles


def test_profiles_inherit_root_defaults_and_keep_option_specific_parameters():
    profiles = build_retrieval_profiles(
        ("children_a", "children_b"),
        dense_vector_name="text-dense",
        sparse_vector_name="text-sparse",
    )

    vector = profiles["simple_vector"]
    sparse = profiles["simple_sparse"]
    hybrid = profiles["hybrid"]

    assert (vector.fetch_top_k, vector.min_score, vector.fallback_top_k) == (10, 0.75, 3)
    assert vector.source_filter is None
    assert vector.dense == DenseRetrievalParams(vector_name="text-dense")
    assert sparse.sparse == SparseRetrievalParams(vector_name="text-sparse")
    assert hybrid.dense == vector.dense
    assert hybrid.sparse == sparse.sparse
    assert hybrid.fusion == FusionRetrievalParams(candidate_top_k=5)


def test_parent_child_profiles_are_derived_from_canonical_profiles():
    profiles = build_retrieval_profiles(("children",))

    assert profiles["simple_vector_parent_child"] == replace(
        profiles["simple_vector"],
        name="simple_vector_parent_child",
        parent_child=ParentChildRetrievalParams(),
    )
    assert profiles["hybrid_parent_child"] == replace(
        profiles["hybrid"],
        name="hybrid_parent_child",
        parent_child=ParentChildRetrievalParams(),
    )


def test_reranker_variants_only_change_model_and_parent_option():
    profiles = build_retrieval_profiles(("children",))
    bge = profiles["hybrid_rerank_bge"]
    qwen = profiles["hybrid_rerank_qwen_4b"]
    parent_qwen = profiles["hybrid_parent_child_rerank_qwen_4b"]

    assert bge.fetch_top_k == 20
    assert bge.fusion == FusionRetrievalParams(candidate_top_k=20)
    assert bge.rerank == RerankRetrievalParams(model="bge-v2-m3")
    assert qwen == replace(
        bge, name="hybrid_rerank_qwen_4b", rerank=RerankRetrievalParams(model="qwen-4b")
    )
    assert parent_qwen == replace(
        qwen,
        name="hybrid_parent_child_rerank_qwen_4b",
        parent_child=ParentChildRetrievalParams(),
    )


def test_agent_mapping_preserves_architecture_default():
    assert AGENT_RETRIEVAL_PROFILES["r_helpdesk"] == "simple_vector"
    assert AGENT_RETRIEVAL_PROFILES["sas_to_r"] is None
