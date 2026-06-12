from __future__ import annotations

from canar.app.retrieval.models import RetrievalHit, RetrievalProfile, RetrievalQuery, SparseVector
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
        top_k=5,
        score_threshold=0.35,
        source_filter="utilitr",
        fallback_top_k=3,
        vector_name="text-sparse",
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


def test_simple_sparse_keeps_top_fallback_when_no_hits_survive_threshold():
    profile = RetrievalProfile(
        name="simple_sparse",
        strategy="simple_sparse",
        collections=("docs",),
        top_k=5,
        score_threshold=2.0,
        fallback_top_k=2,
    )
    adapter = FakeSparseAdapter(
        {
            "docs": [
                RetrievalHit(text="one", collection="docs", score=4.0, score_norm=0),
                RetrievalHit(text="two", collection="docs", score=3.0, score_norm=0),
                RetrievalHit(text="three", collection="docs", score=2.0, score_norm=0),
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
    )
    strategy = SimpleSparseStrategy(profile, FakeSparseAdapter({}))

    try:
        strategy.search(RetrievalQuery(text="question", profile_name="simple_sparse"))
    except ValueError as exc:
        assert "sparse query vector" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
