from __future__ import annotations

from canar.app.retrieval.models import RetrievalHit, RetrievalProfile, RetrievalQuery
from canar.app.retrieval.service import RetrievalService


class FakeEmbedClient:
    def __init__(self):
        self.queries = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 2.0]


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
