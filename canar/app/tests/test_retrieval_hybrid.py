from __future__ import annotations

import pytest

from canar.app.retrieval.fusion.rrf import ReciprocalRankFusion
from canar.app.retrieval.models import (
    DenseRetrievalParams,
    FusionRetrievalParams,
    RerankRetrievalParams,
    RetrievalHit,
    RetrievalProfile,
    RetrievalQuery,
    SparseRetrievalParams,
    SparseVector,
)
from canar.app.retrieval.profiles import build_retrieval_profiles
from canar.app.retrieval.strategies.hybrid import HybridStrategy


class FakeStrategy:
    def __init__(self, hits: list[RetrievalHit]):
        self.hits = hits
        self.queries: list[RetrievalQuery] = []

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        self.queries.append(query)
        return self.hits


class RecordingFusion:
    def __init__(self):
        self.calls: list[tuple[list[list[RetrievalHit]], int, list[float] | None]] = []

    def fuse(
        self,
        ranked_lists: list[list[RetrievalHit]],
        top_k: int,
        weights: list[float] | None = None,
    ) -> list[RetrievalHit]:
        self.calls.append((ranked_lists, top_k, weights))
        return [hit for ranked_list in ranked_lists for hit in ranked_list][:top_k]


def hit(text: str, score: float = 1.0, metadata: dict | None = None) -> RetrievalHit:
    return RetrievalHit(
        text=text,
        collection="docs",
        score=score,
        score_norm=score,
        source="utilitr",
        metadata=metadata or {},
    )


def hybrid_query() -> RetrievalQuery:
    return RetrievalQuery(
        text="exact_table_name",
        profile_name="hybrid",
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
    )


def test_hybrid_profile_is_registered_with_expected_parameters():
    profiles = build_retrieval_profiles(("docs",), sparse_vector_name="text-sparse")

    profile = profiles["hybrid"]

    assert profile.name == "hybrid"
    assert profile.strategy == "hybrid"
    assert profile.collections == ("docs",)
    assert profile.dense == DenseRetrievalParams(
        top_k=10,
        min_score=0.35,
        max_kept=None,
    )
    assert profile.sparse == SparseRetrievalParams(
        top_k=10,
        min_score_ratio=0.35,
        gap_ratio=None,
        max_kept=None,
    )
    assert profile.fusion == FusionRetrievalParams(
        method="rrf",
        rrf_k=60,
        weights={"dense": 1.0, "sparse": 1.0},
        final_top_k=5,
    )

    assert profile.rerank == RerankRetrievalParams(
        candidate_top_k=20,
        rerank_top_n=5,
        reranker_model=None,
    )
    assert profile.vector_name == "text-sparse"


def test_flat_hybrid_profile_fields_are_resolved_for_backward_compatibility():
    profile = RetrievalProfile(
        name="legacy_hybrid",
        strategy="hybrid",
        collections=("docs",),
        top_k=5,
        score_threshold=0.42,
        dense_top_k=30,
        sparse_top_k=20,
        fusion="weighted_rrf",
        final_top_k=8,
        rrf_k=12,
        dense_weight=1.5,
        sparse_weight=2.0,
    )

    assert profile.dense_params() == DenseRetrievalParams(
        top_k=30,
        min_score=0.42,
    )
    assert profile.sparse_params() == SparseRetrievalParams(top_k=20)
    assert profile.fusion_params() == FusionRetrievalParams(
        method="weighted_rrf",
        rrf_k=12,
        weights={"dense": 1.5, "sparse": 2.0},
        final_top_k=8,
    )


def test_flat_rerank_candidate_field_is_resolved_for_backward_compatibility():
    profile = RetrievalProfile(
        name="legacy_hybrid",
        strategy="hybrid",
        collections=("docs",),
        rerank_candidate_top_k=30,
    )

    assert profile.rerank_params() == RerankRetrievalParams(
        candidate_top_k=30,
        rerank_top_n=5,
        reranker_model=None,
    )


def test_simple_vector_profile_uses_dense_vector_name():
    profiles = build_retrieval_profiles(
        ("docs",),
        dense_vector_name="text-dense",
        sparse_vector_name="text-sparse",
    )

    assert profiles["simple_vector"].vector_name == "text-dense"
    assert profiles["simple_sparse"].vector_name == "text-sparse"


def test_rrf_combines_ranked_lists_and_merges_duplicate_hits():
    duplicate_dense = hit("shared document", metadata={"chunk_id": "shared"})
    duplicate_sparse = hit("shared document", metadata={"chunk_id": "shared"})

    fused = ReciprocalRankFusion().fuse(
        [
            [hit("dense only"), duplicate_dense],
            [duplicate_sparse, hit("sparse only")],
        ],
        top_k=10,
    )

    assert [result.text for result in fused] == ["shared document", "dense only", "sparse only"]
    assert len(fused) == 3
    assert fused[0].score_norm == 1.0


def test_rrf_equal_weights_match_default_behavior():
    ranked_lists = [
        [hit("dense top"), hit("dense second")],
        [hit("sparse top"), hit("sparse second")],
    ]

    default_fused = ReciprocalRankFusion().fuse(ranked_lists, top_k=4)
    weighted_fused = ReciprocalRankFusion().fuse(
        ranked_lists,
        top_k=4,
        weights=[1.0, 1.0],
    )

    assert [result.text for result in weighted_fused] == [result.text for result in default_fused]
    assert [result.score for result in weighted_fused] == [result.score for result in default_fused]


def test_rrf_dense_weight_can_favor_dense_results():
    fused = ReciprocalRankFusion().fuse(
        [[hit("dense top")], [hit("sparse top")]],
        top_k=2,
        weights=[2.0, 1.0],
    )

    assert [result.text for result in fused] == ["dense top", "sparse top"]


def test_rrf_sparse_weight_can_favor_sparse_results():
    fused = ReciprocalRankFusion().fuse(
        [[hit("dense top")], [hit("sparse top")]],
        top_k=2,
        weights=[1.0, 2.0],
    )

    assert [result.text for result in fused] == ["sparse top", "dense top"]


def test_hybrid_strategy_uses_query_candidate_top_k_override():
    profile = RetrievalProfile(
        name="hybrid",
        strategy="hybrid",
        collections=("docs",),
        fusion="rrf",
        final_top_k=5,
    )
    dense = FakeStrategy([hit("dense top")])
    sparse = FakeStrategy([hit("sparse top")])
    fusion = RecordingFusion()
    query = RetrievalQuery(
        text="exact_table_name",
        profile_name="hybrid",
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        candidate_top_k=20,
    )

    HybridStrategy(profile, dense, sparse, fusion_strategies={"rrf": fusion}).search(query)

    _ranked_lists, top_k, _weights = fusion.calls[0]
    assert top_k == 20


def test_hybrid_strategy_passes_weights_in_dense_then_sparse_order():
    profile = RetrievalProfile(
        name="hybrid",
        strategy="hybrid",
        collections=("docs",),
        fusion=FusionRetrievalParams(
            method="weighted_rrf",
            rrf_k=60,
            weights={"dense": 3.0, "sparse": 1.0},
            final_top_k=2,
        ),
    )
    dense = FakeStrategy([hit("dense top")])
    sparse = FakeStrategy([hit("sparse top")])
    fusion = RecordingFusion()

    HybridStrategy(profile, dense, sparse, fusion_strategies={"weighted_rrf": fusion}).search(
        hybrid_query()
    )

    ranked_lists, top_k, weights = fusion.calls[0]
    assert ranked_lists == [dense.hits, sparse.hits]
    assert top_k == 2
    assert weights == [3.0, 1.0]


def test_hybrid_strategy_uses_profile_rrf_k():
    profile = RetrievalProfile(
        name="hybrid",
        strategy="hybrid",
        collections=("docs",),
        fusion=FusionRetrievalParams(
            method="rrf",
            rrf_k=2,
            final_top_k=1,
        ),
    )
    dense = FakeStrategy([hit("dense top")])
    sparse = FakeStrategy([])

    hits = HybridStrategy(profile, dense, sparse).search(hybrid_query())

    assert hits[0].score == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    "weights",
    [
        ["heavy"],
        [True],
        [float("inf")],
        [-1.0],
        [0.0],
    ],
)
def test_rrf_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="RRF weight|At least one RRF weight"):
        ReciprocalRankFusion().fuse([[hit("dense top")]], top_k=1, weights=weights)


def test_rrf_rejects_weight_count_mismatch():
    with pytest.raises(ValueError, match="RRF weights must match"):
        ReciprocalRankFusion().fuse(
            [[hit("dense top")], [hit("sparse top")]],
            top_k=2,
            weights=[1.0],
        )


def test_hybrid_strategy_calls_dense_and_sparse_paths_and_limits_results():
    profile = RetrievalProfile(
        name="hybrid",
        strategy="hybrid",
        collections=("docs",),
        fusion=FusionRetrievalParams(method="rrf", final_top_k=2),
    )
    dense = FakeStrategy([hit("dense top"), hit("shared", metadata={"chunk_id": "same"})])
    sparse = FakeStrategy([hit("shared", metadata={"chunk_id": "same"}), hit("sparse exact")])
    query = hybrid_query()

    hits = HybridStrategy(profile, dense, sparse).search(query)

    assert [result.text for result in hits] == ["shared", "dense top"]
    assert len(hits) == 2
    assert dense.queries == [
        RetrievalQuery(
            text="exact_table_name",
            profile_name="hybrid:dense",
            dense_vector=[0.1, 0.2],
            sparse_vector=SparseVector(indices=[1], values=[1.0]),
        )
    ]
    assert sparse.queries == [
        RetrievalQuery(
            text="exact_table_name",
            profile_name="hybrid:sparse",
            dense_vector=[0.1, 0.2],
            sparse_vector=SparseVector(indices=[1], values=[1.0]),
        )
    ]


def test_hybrid_strategy_keeps_sparse_exact_match_in_fused_results():
    profile = RetrievalProfile(
        name="hybrid",
        strategy="hybrid",
        collections=("docs",),
        fusion=FusionRetrievalParams(method="rrf", final_top_k=3),
    )
    dense = FakeStrategy([hit("semantic result one"), hit("semantic result two")])
    sparse = FakeStrategy([hit("exact variable_name result")])
    query = RetrievalQuery(
        text="variable_name",
        profile_name="hybrid",
        dense_vector=[0.1],
        sparse_vector=SparseVector(indices=[4], values=[0.8]),
    )

    hits = HybridStrategy(profile, dense, sparse).search(query)

    assert "exact variable_name result" in [result.text for result in hits]


def test_hybrid_strategy_rejects_unknown_fusion_method():
    profile = RetrievalProfile(
        name="hybrid",
        strategy="hybrid",
        collections=("docs",),
        fusion="unknown",
    )
    query = RetrievalQuery(
        text="question",
        profile_name="hybrid",
        dense_vector=[0.1],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
    )

    with pytest.raises(ValueError, match="Unsupported hybrid fusion strategy"):
        HybridStrategy(profile, FakeStrategy([]), FakeStrategy([])).search(query)
