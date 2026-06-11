from __future__ import annotations

from canar.app.retrieval.models import RetrievalHit, RetrievalProfile, RetrievalQuery
from canar.app.retrieval.strategies.simple_vector import SimpleVectorStrategy


class FakeDenseAdapter:
    def __init__(self, hits_by_collection: dict[str, list[RetrievalHit]]):
        self.hits_by_collection = hits_by_collection
        self.calls = []

    def search_dense(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        source_filter: str | None = "utilitr",
        vector_name: str | None = None,
    ) -> list[RetrievalHit]:
        self.calls.append((collection, query_vector, top_k, source_filter, vector_name))
        return self.hits_by_collection.get(collection, [])


def test_simple_vector_normalizes_sorts_and_prunes_hits():
    profile = RetrievalProfile(
        name="simple_vector",
        strategy="simple_vector",
        collections=("docs_a", "docs_b"),
        top_k=5,
        score_threshold=0.35,
        source_filter="utilitr",
        fallback_top_k=3,
    )
    adapter = FakeDenseAdapter(
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

    hits = SimpleVectorStrategy(profile, adapter).search(
        RetrievalQuery(text="question", profile_name="simple_vector", dense_vector=[0.1, 0.2])
    )

    assert [hit.text for hit in hits] == ["high"]
    assert hits[0].score_norm == 1.0
    assert adapter.calls == [
        ("docs_a", [0.1, 0.2], 5, "utilitr", None),
        ("docs_b", [0.1, 0.2], 5, "utilitr", None),
    ]


def test_simple_vector_keeps_top_fallback_when_all_hits_are_below_threshold():
    profile = RetrievalProfile(
        name="simple_vector",
        strategy="simple_vector",
        collections=("docs",),
        top_k=5,
        score_threshold=2.0,
        source_filter="utilitr",
        fallback_top_k=3,
    )
    adapter = FakeDenseAdapter(
        {
            "docs": [
                RetrievalHit(text="one", collection="docs", score=4.0, score_norm=0),
                RetrievalHit(text="two", collection="docs", score=3.0, score_norm=0),
                RetrievalHit(text="three", collection="docs", score=2.0, score_norm=0),
                RetrievalHit(text="four", collection="docs", score=1.0, score_norm=0),
            ]
        }
    )

    hits = SimpleVectorStrategy(profile, adapter).search(
        RetrievalQuery(text="question", profile_name="simple_vector", dense_vector=[0.1])
    )

    assert [hit.text for hit in hits] == ["one", "two", "three"]


def test_simple_vector_requires_dense_vector():
    profile = RetrievalProfile(
        name="simple_vector",
        strategy="simple_vector",
        collections=("docs",),
    )
    strategy = SimpleVectorStrategy(profile, FakeDenseAdapter({}))

    try:
        strategy.search(RetrievalQuery(text="question", profile_name="simple_vector"))
    except ValueError as exc:
        assert "dense query vector" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
