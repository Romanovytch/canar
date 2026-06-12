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
