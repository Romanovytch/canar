from __future__ import annotations

from canar.app.retrieval.fusion.rrf import ReciprocalRankFusion
from canar.app.retrieval.models import RetrievalHit, RetrievalProfile, RetrievalQuery, SparseVector
from canar.app.retrieval.profiles import build_retrieval_profiles
from canar.app.retrieval.strategies.hybrid import HybridStrategy


class FakeStrategy:
    def __init__(self, hits: list[RetrievalHit]):
        self.hits = hits
        self.queries: list[RetrievalQuery] = []

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        self.queries.append(query)
        return self.hits


def hit(text: str, score: float = 1.0, metadata: dict | None = None) -> RetrievalHit:
    return RetrievalHit(
        text=text,
        collection="docs",
        score=score,
        score_norm=score,
        source="utilitr",
        metadata=metadata or {},
    )


def test_hybrid_profile_is_registered_with_expected_parameters():
    profiles = build_retrieval_profiles(("docs",), sparse_vector_name="text-sparse")

    profile = profiles["hybrid"]

    assert profile.name == "hybrid"
    assert profile.strategy == "hybrid"
    assert profile.collections == ("docs",)
    assert profile.dense_top_k == 10
    assert profile.sparse_top_k == 10
    assert profile.fusion == "rrf"
    assert profile.final_top_k == 5
    assert profile.vector_name == "text-sparse"


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


def test_hybrid_strategy_calls_dense_and_sparse_paths_and_limits_results():
    profile = RetrievalProfile(
        name="hybrid",
        strategy="hybrid",
        collections=("docs",),
        fusion="rrf",
        final_top_k=2,
    )
    dense = FakeStrategy([hit("dense top"), hit("shared", metadata={"chunk_id": "same"})])
    sparse = FakeStrategy([hit("shared", metadata={"chunk_id": "same"}), hit("sparse exact")])
    query = RetrievalQuery(
        text="exact_table_name",
        profile_name="hybrid",
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
    )

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
        fusion="rrf",
        final_top_k=3,
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

    try:
        HybridStrategy(profile, FakeStrategy([]), FakeStrategy([])).search(query)
    except ValueError as exc:
        assert "Unsupported hybrid fusion strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
