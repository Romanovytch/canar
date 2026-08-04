from __future__ import annotations

from types import SimpleNamespace

import pytest

from canar.app.retrieval.adapters.qdrant import QdrantRetrievalAdapter
from canar.app.retrieval.models import (
    DenseRetrievalParams,
    FusionRetrievalParams,
    ParentChildRetrievalParams,
    RetrievalHit,
    RetrievalProfile,
    SparseRetrievalParams,
    SparseVector,
    SummaryRetrievalParams,
)
from canar.app.retrieval.profiles import (
    AGENT_RETRIEVAL_PROFILES,
    build_retrieval_profiles,
)
from canar.app.retrieval.service import RetrievalService


class FakeEmbedClient:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 2.0]


class FakeSparseEmbedClient:
    def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[3], values=[0.7])


class FakeConfig:
    qdrant_collections = ("docs_a", "docs_b")
    qdrant_dense_vector_name = "text-dense"
    qdrant_sparse_vector_name = "text-sparse"
    fastembed_sparse_model = ""
    qdrant_url = "http://qdrant.test"
    qdrant_api_key = ""


class RecordingQdrantAdapter:
    instances: list[RecordingQdrantAdapter] = []
    return_dense = True
    return_sparse = True

    def __init__(self, url: str, api_key: str | None = None):
        self.dense_calls = []
        self.sparse_calls = []
        self.instances.append(self)

    def search_dense(self, **kwargs) -> list[RetrievalHit]:
        self.dense_calls.append(kwargs)
        if not self.return_dense:
            return []
        collection = kwargs["collection"]
        return [
            RetrievalHit(
                text=f"dense {collection}",
                collection=collection,
                score=0.9,
                score_norm=0.0,
                metadata={"id": f"dense-{collection}"},
            )
        ]

    def search_sparse(self, **kwargs) -> list[RetrievalHit]:
        self.sparse_calls.append(kwargs)
        if not self.return_sparse:
            return []
        collection = kwargs["collection"]
        return [
            RetrievalHit(
                text=f"sparse {collection}",
                collection=collection,
                score=0.8,
                score_norm=0.0,
                metadata={"id": f"sparse-{collection}"},
            )
        ]


def test_summary_params_require_hybrid_non_empty_suffix_and_no_parent():
    with pytest.raises(ValueError, match="requires strategy='hybrid'"):
        RetrievalProfile(
            name="invalid",
            strategy="simple_vector",
            collections=("docs",),
            dense=DenseRetrievalParams(),
            summary=SummaryRetrievalParams(collection_suffix="_summaries"),
        )

    with pytest.raises(ValueError, match="requires a non-empty collection suffix"):
        RetrievalProfile(
            name="invalid",
            strategy="hybrid",
            collections=("docs",),
            dense=DenseRetrievalParams(),
            sparse=SparseRetrievalParams(),
            fusion=FusionRetrievalParams(),
            summary=SummaryRetrievalParams(collection_suffix="  "),
        )

    with pytest.raises(ValueError, match="cannot be combined with parent retrieval"):
        RetrievalProfile(
            name="invalid",
            strategy="hybrid",
            collections=("docs",),
            dense=DenseRetrievalParams(),
            sparse=SparseRetrievalParams(),
            fusion=FusionRetrievalParams(),
            summary=SummaryRetrievalParams(collection_suffix="_summaries"),
            parent_child=ParentChildRetrievalParams(),
        )


def test_summaries_profile_is_registered_and_resolves_collections():
    profiles = build_retrieval_profiles(("docs_a", "docs_b"))

    assert AGENT_RETRIEVAL_PROFILES["generic_agent"] == "hybrid_summary"
    summary = profiles["hybrid_summary"]
    assert summary.summary == SummaryRetrievalParams(collection_suffix="_summaries")
    assert summary.fusion == FusionRetrievalParams(
        weights={"dense": 3.0, "sparse": 1.0},
        output_top_k=2,
    )
    assert summary.effective_collections() == (
        "docs_a_summary",
        "docs_b_summary",
    )
    assert profiles["hybrid"].effective_collections() == ("docs_a", "docs_b")


def test_service_runs_both_hybrid_paths_against_summary_collections(monkeypatch):
    import canar.app.retrieval.service as service_module

    RecordingQdrantAdapter.instances.clear()
    RecordingQdrantAdapter.return_dense = True
    RecordingQdrantAdapter.return_sparse = True
    monkeypatch.setattr(
        service_module,
        "QdrantRetrievalAdapter",
        RecordingQdrantAdapter,
    )
    service = RetrievalService.from_config(
        FakeConfig(),
        embed_client=FakeEmbedClient(),
        sparse_embed_client=FakeSparseEmbedClient(),
        agent_profiles={"generic_agent": "hybrid_summary"},
    )

    hits = service.search("generic_agent", "question")

    adapter = RecordingQdrantAdapter.instances[0]
    expected_collections = ["docs_a_summary", "docs_b_summary"]
    assert [call["collection"] for call in adapter.dense_calls] == expected_collections
    assert [call["collection"] for call in adapter.sparse_calls] == expected_collections
    assert {call["vector_name"] for call in adapter.dense_calls} == {"text-dense"}
    assert {call["vector_name"] for call in adapter.sparse_calls} == {"text-sparse"}
    assert {call["source_filter"] for call in adapter.dense_calls} == {None}
    assert {call["source_filter"] for call in adapter.sparse_calls} == {None}
    assert {hit.collection for hit in hits} <= set(expected_collections)
    fusion = service.profiles["hybrid_summary"].fusion
    assert fusion is not None
    assert fusion.output_top_k is not None
    assert len(hits) <= fusion.output_top_k


@pytest.mark.parametrize(
    ("return_dense", "return_sparse", "expects_hits"),
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_summary_hybrid_supports_one_or_both_empty_paths(
    monkeypatch,
    return_dense: bool,
    return_sparse: bool,
    expects_hits: bool,
):
    import canar.app.retrieval.service as service_module

    RecordingQdrantAdapter.instances.clear()
    RecordingQdrantAdapter.return_dense = return_dense
    RecordingQdrantAdapter.return_sparse = return_sparse
    monkeypatch.setattr(
        service_module,
        "QdrantRetrievalAdapter",
        RecordingQdrantAdapter,
    )
    service = RetrievalService.from_config(
        FakeConfig(),
        embed_client=FakeEmbedClient(),
        sparse_embed_client=FakeSparseEmbedClient(),
        agent_profiles={"generic_agent": "hybrid_summary"},
    )

    hits = service.search("generic_agent", "question")

    assert bool(hits) is expects_hits


def test_dense_qdrant_failure_includes_collection_and_vector_name():
    class FailingClient:
        def query_points(self, **kwargs):
            raise OSError("backend unavailable")

    adapter = QdrantRetrievalAdapter.__new__(QdrantRetrievalAdapter)
    adapter.client = FailingClient()

    with pytest.raises(
        RuntimeError,
        match="Dense retrieval failed for collection 'docs_summary'.*'text-dense'",
    ):
        adapter.search_dense(
            collection="docs_summary",
            query_vector=[1.0],
            top_k=5,
            vector_name="text-dense",
        )


def test_qdrant_adapter_converts_summary_payload_to_retrieval_hit():
    point = SimpleNamespace(
        score=0.91,
        payload={
            "text": "Condensed document",
            "source": "utilitr",
            "url": "https://example.test/doc",
            "section": "Résumé",
            "document_id": "doc-1",
        },
    )

    class SuccessfulClient:
        def query_points(self, **kwargs):
            return SimpleNamespace(points=[point])

    adapter = QdrantRetrievalAdapter.__new__(QdrantRetrievalAdapter)
    adapter.client = SuccessfulClient()

    hits = adapter.search_dense(
        collection="docs_summary",
        query_vector=[1.0],
        top_k=5,
    )

    assert hits == [
        RetrievalHit(
            text="Condensed document",
            collection="docs_summary",
            score=0.91,
            score_norm=0.0,
            source="utilitr",
            source_url="https://example.test/doc",
            section="Résumé",
            metadata=point.payload,
        )
    ]
