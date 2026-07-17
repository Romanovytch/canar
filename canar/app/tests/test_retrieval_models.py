from __future__ import annotations

import pytest

from canar.app.retrieval.models import (
    DenseRetrievalParams,
    FusionRetrievalParams,
    RetrievalProfile,
    SparseRetrievalParams,
)


@pytest.mark.parametrize(
    ("strategy", "expected_missing"),
    [
        ("simple_vector", "dense"),
        ("simple_sparse", "sparse"),
        ("hybrid", "dense, sparse, fusion"),
    ],
)
def test_profile_validates_required_strategy_blocks(strategy, expected_missing):
    with pytest.raises(ValueError, match=expected_missing):
        RetrievalProfile(name="invalid", strategy=strategy, collections=("docs",))


def test_profile_accepts_complete_hybrid_configuration():
    profile = RetrievalProfile(
        name="hybrid",
        strategy="hybrid",
        collections=("docs",),
        dense=DenseRetrievalParams(),
        sparse=SparseRetrievalParams(),
        fusion=FusionRetrievalParams(),
    )

    assert profile.fetch_top_k == 10
    assert profile.min_score == 0.75
    assert profile.fallback_top_k == 3
    assert profile.source_filter is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fetch_top_k", 0),
        ("min_score", 1.1),
        ("fallback_top_k", -1),
        ("max_results", 0),
    ],
)
def test_profile_rejects_invalid_root_policy(field, value):
    kwargs = {
        "name": "simple_vector",
        "strategy": "simple_vector",
        "collections": ("docs",),
        "dense": DenseRetrievalParams(),
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        RetrievalProfile(**kwargs)


@pytest.mark.parametrize(
    "params",
    [
        DenseRetrievalParams(fetch_top_k=4, min_score=0.4),
        SparseRetrievalParams(fetch_top_k=6, min_score=0.6, gap_ratio=0.2),
    ],
)
def test_retriever_params_accept_valid_policy_overrides(params):
    assert params.fetch_top_k is not None
    assert params.min_score is not None


@pytest.mark.parametrize(
    "params_type, kwargs, expected",
    [
        (DenseRetrievalParams, {"fetch_top_k": 0}, "fetch_top_k"),
        (SparseRetrievalParams, {"min_score": 1.1}, "min_score"),
        (SparseRetrievalParams, {"gap_ratio": -0.1}, "gap_ratio"),
    ],
)
def test_retriever_params_reject_invalid_policy_overrides(params_type, kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        params_type(**kwargs)
