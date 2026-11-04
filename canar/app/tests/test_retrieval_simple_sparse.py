from __future__ import annotations

from canar.app.retrieval.models import (
    RetrievalHit,
    RetrievalProfile,
    RetrievalQuery,
    SparseRetrievalParams,
    SparseVector,
)
from canar.app.retrieval.strategies.simple_sparse import SimpleSparseStrategy


class FakeSparseAdapter:
    def __init__(self, hits_by_collection: dict[str, list[RetrievalHit]]):
        self.hits_by_collection = hits_by_collection
        self.calls = []

    def search_sparse(
        self,
        collection: str,
        query_vector: SparseVector,
        top_k: int,
        source_filter: str | None = "utilitr",
        vector_name: str | None = None,
    ) -> list[RetrievalHit]:
        self.calls.append((collection, query_vector, top_k, source_filter, vector_name))
        return self.hits_by_collection.get(collection, [])


def test_simple_sparse_normalizes_sorts_and_prunes_hits():
    profile = RetrievalProfile(
        name="simple_sparse",
        strategy="simple_sparse",
        collections=("docs_a", "docs_b"),
        fetch_top_k=5,
        min_score=0.35,
        sparse=SparseRetrievalParams(vector_name="text-sparse"),
        source_filter="utilitr",
        fallback_top_k=3,
    )
    adapter = FakeSparseAdapter(
        {
            "docs_a": [
                RetrievalHit(text="low", collection="docs_a", score=0.2, score_norm=0),
                RetrievalHit(text="high", collection="docs_a", score=0.6, score_norm=0),
            ],
            "docs_b": [
                RetrievalHit(text="flat 1", collection="docs_b", score=10.0, score_norm=0),
                RetrievalHit(text="flat 2", collection="docs_b", score=10.0, score_norm=0),
            ],
        }
    )
    sparse_vector = SparseVector(indices=[1, 4], values=[0.3, 0.7])

    hits = SimpleSparseStrategy(profile, adapter).search(
        RetrievalQuery(
            text="question",
            profile_name="simple_sparse",
            sparse_vector=sparse_vector,
        )
    )

    assert [hit.text for hit in hits] == ["high"]
    assert hits[0].score_norm == 1.0
    assert adapter.calls == [
        ("docs_a", sparse_vector, 5, "utilitr", "text-sparse"),
        ("docs_b", sparse_vector, 5, "utilitr", "text-sparse"),
    ]


def test_simple_sparse_params_override_profile_policy_and_apply_gap_ratio():
    profile = RetrievalProfile(
        name="hybrid_sparse",
        strategy="simple_sparse",
        collections=("docs",),
        fetch_top_k=10,
        min_score=0.75,
        sparse=SparseRetrievalParams(
            vector_name="text-sparse",
            fetch_top_k=4,
            min_score=0.5,
            gap_ratio=0.25,
        ),
    )
    adapter = FakeSparseAdapter(
        {
            "docs": [
                RetrievalHit(text="top", collection="docs", score=10.0, score_norm=0),
                RetrievalHit(text="second", collection="docs", score=8.0, score_norm=0),
                RetrievalHit(text="third", collection="docs", score=6.0, score_norm=0),
                RetrievalHit(text="low", collection="docs", score=1.0, score_norm=0),
            ]
        }
    )
    sparse_vector = SparseVector(indices=[1], values=[1.0])

    hits = SimpleSparseStrategy(profile, adapter).search(
        RetrievalQuery(
            text="question",
            profile_name="hybrid_sparse",
            sparse_vector=sparse_vector,
        )
    )

    assert [hit.text for hit in hits] == ["top", "second"]
    assert adapter.calls == [("docs", sparse_vector, 4, None, "text-sparse")]


def test_simple_sparse_keeps_top_fallback_when_no_hits_survive_threshold():
    profile = RetrievalProfile(
        name="simple_sparse",
        strategy="simple_sparse",
        collections=("docs",),
        fetch_top_k=5,
        min_score=0.75,
        sparse=SparseRetrievalParams(),
        fallback_top_k=2,
    )
    adapter = FakeSparseAdapter(
        {
            "docs": [
                RetrievalHit(text="one", collection="docs", score=1.0, score_norm=0),
                RetrievalHit(text="two", collection="docs", score=1.0, score_norm=0),
                RetrievalHit(text="three", collection="docs", score=1.0, score_norm=0),
            ]
        }
    )

    hits = SimpleSparseStrategy(profile, adapter).search(
        RetrievalQuery(
            text="question",
            profile_name="simple_sparse",
            sparse_vector=SparseVector(indices=[1], values=[1.0]),
        )
    )

    assert [hit.text for hit in hits] == ["one", "two"]


def test_simple_sparse_requires_sparse_vector():
    profile = RetrievalProfile(
        name="simple_sparse",
        strategy="simple_sparse",
        collections=("docs",),
        sparse=SparseRetrievalParams(),
    )
    strategy = SimpleSparseStrategy(profile, FakeSparseAdapter({}))

    try:
        strategy.search(RetrievalQuery(text="question", profile_name="simple_sparse"))
    except ValueError as exc:
        assert "sparse query vector" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
