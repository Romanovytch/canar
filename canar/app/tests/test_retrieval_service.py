from __future__ import annotations

from canar.app.retrieval.models import RetrievalHit, RetrievalProfile, RetrievalQuery, SparseVector
from canar.app.retrieval.service import RetrievalService


class FakeEmbedClient:
    def __init__(self):
        self.queries = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 2.0]


class FakeSparseEmbedClient:
    def __init__(self):
        self.queries = []

    def embed_query(self, text: str) -> SparseVector:
        self.queries.append(text)
        return SparseVector(indices=[3], values=[0.7])


class FakeConfig:
    qdrant_collections = ("docs",)
    qdrant_dense_vector_name = "text-dense"
    qdrant_sparse_vector_name = "text-sparse"
    fastembed_sparse_model = ""
    qdrant_url = "http://qdrant.test"
    qdrant_api_key = ""


class FakeStrategy:
    def __init__(self):
        self.queries: list[RetrievalQuery] = []

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        self.queries.append(query)
        return [
            RetrievalHit(
                text="answer context",
                collection="docs",
                score=1.0,
                score_norm=1.0,
            )
        ]


def test_retrieval_service_selects_profile_by_agent_and_embeds_query():
    embed = FakeEmbedClient()
    strategy = FakeStrategy()
    service = RetrievalService(
        embed_client=embed,
        profiles={
            "simple_vector": RetrievalProfile(
                name="simple_vector",
                strategy="simple_vector",
                collections=("docs",),
            )
        },
        agent_profiles={"r_helpdesk": "simple_vector"},
        strategies={"simple_vector": strategy},
    )

    hits = service.search("r_helpdesk", "Comment filtrer un dataframe ?")

    assert [hit.text for hit in hits] == ["answer context"]
    assert embed.queries == ["Comment filtrer un dataframe ?"]
    assert strategy.queries == [
        RetrievalQuery(
            text="Comment filtrer un dataframe ?",
            profile_name="simple_vector",
            dense_vector=[1.0, 2.0],
        )
    ]


def test_retrieval_service_returns_no_hits_for_agent_without_profile():
    embed = FakeEmbedClient()
    service = RetrievalService(
        embed_client=embed,
        profiles={},
        agent_profiles={"sas_to_r": None},
        strategies={},
    )

    assert service.search("sas_to_r", "proc sort") == []
    assert embed.queries == []


def test_retrieval_service_embeds_sparse_only_for_sparse_profile():
    dense_embed = FakeEmbedClient()
    sparse_embed = FakeSparseEmbedClient()
    strategy = FakeStrategy()
    service = RetrievalService(
        embed_client=dense_embed,
        sparse_embed_client=sparse_embed,
        profiles={
            "simple_sparse": RetrievalProfile(
                name="simple_sparse",
                strategy="simple_sparse",
                collections=("docs",),
            )
        },
        agent_profiles={"r_helpdesk": "simple_sparse"},
        strategies={"simple_sparse": strategy},
    )

    hits = service.search("r_helpdesk", "Comment joindre deux tables ?")

    assert [hit.text for hit in hits] == ["answer context"]
    assert dense_embed.queries == []
    assert sparse_embed.queries == ["Comment joindre deux tables ?"]
    assert strategy.queries == [
        RetrievalQuery(
            text="Comment joindre deux tables ?",
            profile_name="simple_sparse",
            sparse_vector=SparseVector(indices=[3], values=[0.7]),
        )
    ]


def test_retrieval_service_embeds_dense_and_sparse_for_hybrid_profile():
    dense_embed = FakeEmbedClient()
    sparse_embed = FakeSparseEmbedClient()
    strategy = FakeStrategy()
    service = RetrievalService(
        embed_client=dense_embed,
        sparse_embed_client=sparse_embed,
        profiles={
            "hybrid": RetrievalProfile(
                name="hybrid",
                strategy="hybrid",
                collections=("docs",),
                dense_top_k=30,
                sparse_top_k=30,
                fusion="rrf",
                final_top_k=10,
            )
        },
        agent_profiles={"r_helpdesk": "hybrid"},
        strategies={"hybrid": strategy},
    )

    hits = service.search("r_helpdesk", "table filtre code_exact")

    assert [hit.text for hit in hits] == ["answer context"]
    assert dense_embed.queries == ["table filtre code_exact"]
    assert sparse_embed.queries == ["table filtre code_exact"]
    assert strategy.queries == [
        RetrievalQuery(
            text="table filtre code_exact",
            profile_name="hybrid",
            dense_vector=[1.0, 2.0],
            sparse_vector=SparseVector(indices=[3], values=[0.7]),
        )
    ]


def test_retrieval_service_builds_hybrid_with_dense_and_sparse_vector_names(
    monkeypatch,
):
    class FakeQdrantAdapter:
        def __init__(self, url: str, api_key: str | None = None):
            self.url = url
            self.api_key = api_key

    import canar.app.retrieval.service as service_module

    monkeypatch.setattr(service_module, "QdrantRetrievalAdapter", FakeQdrantAdapter)

    service = RetrievalService.from_config(
        FakeConfig(),
        embed_client=FakeEmbedClient(),
        sparse_embed_client=FakeSparseEmbedClient(),
    )

    hybrid = service.strategies["hybrid"]

    assert service.strategies["simple_vector"].profile.vector_name == "text-dense"
    assert hybrid.dense_strategy.profile.vector_name == "text-dense"
    assert hybrid.sparse_strategy.profile.vector_name == "text-sparse"
    assert hybrid.dense_strategy.profile.top_k == 10
    assert hybrid.sparse_strategy.profile.top_k == 10
