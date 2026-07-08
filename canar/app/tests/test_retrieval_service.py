from __future__ import annotations

from canar.app.retrieval.models import (
    DenseRetrievalParams,
    ParentChildRetrievalParams,
    RerankRetrievalParams,
    RetrievalHit,
    RetrievalProfile,
    RetrievalQuery,
    SparseRetrievalParams,
    SparseVector,
)
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
    rerank_enabled = False
    reranker_name = "bge"
    rerank_model_name = ""
    rerank_top_n = 5
    rerank_device = "cuda"
    rerank_max_length = 8192
    qdrant_url = "http://qdrant.test"
    qdrant_api_key = ""


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalHit],
        top_n: int | None = None,
    ) -> list[RetrievalHit]:
        self.calls.append((query, candidates, top_n))
        reranked = list(reversed(candidates))
        if top_n is not None:
            return reranked[:top_n]
        return reranked


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


class FakeExpander:
    def __init__(self):
        self.hits: list[list[RetrievalHit]] = []
        self.params: list[ParentChildRetrievalParams] = []

    def expand(
        self,
        hits: list[RetrievalHit],
        params: ParentChildRetrievalParams,
    ) -> list[RetrievalHit]:
        self.hits.append(hits)
        self.params.append(params)
        return [
            RetrievalHit(
                text=hit.text,
                generation_text="expanded context",
                collection=hit.collection,
                score=hit.score,
                score_norm=hit.score_norm,
            )
            for hit in hits
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


def test_retrieval_service_expands_hits_when_profile_enables_expansion():
    embed = FakeEmbedClient()
    strategy = FakeStrategy()
    expander = FakeExpander()
    service = RetrievalService(
        embed_client=embed,
        profiles={
            "simple_vector_parent_child": RetrievalProfile(
                name="simple_vector_parent_child",
                strategy="simple_vector",
                collections=("docs",),
                parent_child=ParentChildRetrievalParams(
                    parent_collection_suffix="_parents",
                ),
            )
        },
        agent_profiles={"r_helpdesk": "simple_vector_parent_child"},
        strategies={"simple_vector": strategy},
        hit_expanders={"parent_child": expander},
    )

    hits = service.search("r_helpdesk", "Comment filtrer un dataframe ?")

    assert [hit.generation_text for hit in hits] == ["expanded context"]
    assert len(expander.hits) == 1
    assert [hit.text for hit in expander.hits[0]] == ["answer context"]
    assert expander.params == [ParentChildRetrievalParams(parent_collection_suffix="_parents")]


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
    assert hybrid.dense_strategy.profile.dense == DenseRetrievalParams(
        top_k=10,
        min_score=0.35,
        max_kept=None,
    )
    assert hybrid.sparse_strategy.profile.sparse == SparseRetrievalParams(
        top_k=10,
        min_score_ratio=0.35,
        gap_ratio=None,
        max_kept=None,
    )


def test_retrieval_service_from_config_builds_profile_reranker_when_config_disabled(
    monkeypatch,
):
    class FakeQdrantAdapter:
        def __init__(self, url: str, api_key: str | None = None):
            self.url = url
            self.api_key = api_key

    import canar.app.retrieval.service as service_module

    built = []

    def fake_build_reranker(
        name: str,
        *,
        device: str | None,
        max_length: int,
        model_name: str | None = None,
    ):
        built.append((name, device, max_length, model_name))
        return FakeReranker()

    monkeypatch.setattr(service_module, "QdrantRetrievalAdapter", FakeQdrantAdapter)
    monkeypatch.setattr(service_module, "build_reranker", fake_build_reranker)

    service = RetrievalService.from_config(
        FakeConfig(),
        embed_client=FakeEmbedClient(),
        sparse_embed_client=FakeSparseEmbedClient(),
    )

    assert service.rerank_enabled is False
    assert service.reranker is not None
    assert built == [("bge", "cuda", 8192, None)]


def test_retrieval_service_from_config_builds_enabled_reranker_with_model_override(
    monkeypatch,
):
    class FakeQdrantAdapter:
        def __init__(self, url: str, api_key: str | None = None):
            self.url = url
            self.api_key = api_key

    class RerankConfig(FakeConfig):
        rerank_enabled = True
        rerank_model_name = "custom/bge-reranker"

    import canar.app.retrieval.service as service_module

    built = []
    fake_reranker = FakeReranker()

    def fake_build_reranker(
        name: str,
        *,
        device: str | None,
        max_length: int,
        model_name: str | None = None,
    ):
        built.append((name, device, max_length, model_name))
        return fake_reranker

    monkeypatch.setattr(service_module, "QdrantRetrievalAdapter", FakeQdrantAdapter)
    monkeypatch.setattr(service_module, "build_reranker", fake_build_reranker)

    service = RetrievalService.from_config(
        RerankConfig(),
        embed_client=FakeEmbedClient(),
        sparse_embed_client=FakeSparseEmbedClient(),
    )

    assert service.rerank_enabled is True
    assert service.reranker is fake_reranker
    assert built == [("bge", "cuda", 8192, "custom/bge-reranker")]


def test_retrieval_service_reranks_hybrid_hits_when_enabled():
    dense_embed = FakeEmbedClient()
    sparse_embed = FakeSparseEmbedClient()
    strategy = FakeStrategy()
    strategy_queries = []

    def search_with_two_hits(query: RetrievalQuery) -> list[RetrievalHit]:
        strategy_queries.append(query)
        return [
            RetrievalHit(text="first", collection="docs", score=1.0, score_norm=1.0),
            RetrievalHit(text="second", collection="docs", score=0.9, score_norm=0.9),
        ]

    strategy.search = search_with_two_hits
    reranker = FakeReranker()
    service = RetrievalService(
        embed_client=dense_embed,
        sparse_embed_client=sparse_embed,
        profiles={
            "hybrid": RetrievalProfile(
                name="hybrid",
                strategy="hybrid",
                collections=("docs",),
                rerank=RerankRetrievalParams(
                    candidate_top_k=20,
                    rerank_top_n=1,
                    reranker_model=None,
                ),
            )
        },
        agent_profiles={"r_helpdesk": "hybrid"},
        strategies={"hybrid": strategy},
        reranker=reranker,
        rerank_enabled=False,
    )

    hits = service.search("r_helpdesk", "comment filtrer ?")

    assert [hit.text for hit in hits] == ["second"]
    assert strategy_queries == [
        RetrievalQuery(
            text="comment filtrer ?",
            profile_name="hybrid",
            dense_vector=[1.0, 2.0],
            sparse_vector=SparseVector(indices=[3], values=[0.7]),
            candidate_top_k=20,
        )
    ]
    assert reranker.calls == [
        (
            "comment filtrer ?",
            [
                RetrievalHit(text="first", collection="docs", score=1.0, score_norm=1.0),
                RetrievalHit(text="second", collection="docs", score=0.9, score_norm=0.9),
            ],
            1,
        )
    ]


def test_retrieval_service_keeps_hybrid_only_when_rerank_is_disabled():
    dense_embed = FakeEmbedClient()
    sparse_embed = FakeSparseEmbedClient()
    strategy = FakeStrategy()
    reranker = FakeReranker()
    service = RetrievalService(
        embed_client=dense_embed,
        sparse_embed_client=sparse_embed,
        profiles={
            "hybrid": RetrievalProfile(
                name="hybrid",
                strategy="hybrid",
                collections=("docs",),
            )
        },
        agent_profiles={"r_helpdesk": "hybrid"},
        strategies={"hybrid": strategy},
        reranker=reranker,
        rerank_enabled=False,
    )

    hits = service.search("r_helpdesk", "comment filtrer ?")

    assert [hit.text for hit in hits] == ["answer context"]
    assert reranker.calls == []


def test_retrieval_service_per_call_rerank_false_overrides_enabled_default():
    dense_embed = FakeEmbedClient()
    sparse_embed = FakeSparseEmbedClient()
    strategy = FakeStrategy()
    reranker = FakeReranker()
    service = RetrievalService(
        embed_client=dense_embed,
        sparse_embed_client=sparse_embed,
        profiles={
            "hybrid": RetrievalProfile(
                name="hybrid",
                strategy="hybrid",
                collections=("docs",),
                rerank=RerankRetrievalParams(),
            )
        },
        agent_profiles={"r_helpdesk": "hybrid"},
        strategies={"hybrid": strategy},
        reranker=reranker,
        rerank_enabled=True,
    )

    hits = service.search("r_helpdesk", "comment filtrer ?", rerank=False)

    assert [hit.text for hit in hits] == ["answer context"]
    assert reranker.calls == []


def test_retrieval_service_per_call_rerank_true_overrides_disabled_default():
    dense_embed = FakeEmbedClient()
    sparse_embed = FakeSparseEmbedClient()
    strategy = FakeStrategy()
    reranker = FakeReranker()
    service = RetrievalService(
        embed_client=dense_embed,
        sparse_embed_client=sparse_embed,
        profiles={
            "hybrid": RetrievalProfile(
                name="hybrid",
                strategy="hybrid",
                collections=("docs",),
                rerank=RerankRetrievalParams(),
            )
        },
        agent_profiles={"r_helpdesk": "hybrid"},
        strategies={"hybrid": strategy},
        reranker=reranker,
        rerank_enabled=False,
    )

    hits = service.search("r_helpdesk", "comment filtrer ?", rerank=True)

    assert [hit.text for hit in hits] == ["answer context"]
    assert len(reranker.calls) == 1


def test_retrieval_service_builds_parent_child_profiles_from_structured_params(
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

    vector_parent_child = service.profiles["simple_vector_parent_child"]
    sparse_parent_child = service.profiles["simple_sparse_parent_child"]
    hybrid_parent_child = service.profiles["hybrid_parent_child"]

    assert set(service.strategies) == {"simple_vector", "simple_sparse", "hybrid"}
    assert "parent_child" in service.hit_expanders
    assert vector_parent_child.strategy == "simple_vector"
    assert vector_parent_child.parent_child == ParentChildRetrievalParams(
        parent_collection_suffix="_parent",
    )
    assert vector_parent_child.dense == DenseRetrievalParams(
        top_k=5,
        min_score=0.35,
        max_kept=None,
    )
    assert sparse_parent_child.strategy == "simple_sparse"
    assert sparse_parent_child.parent_child == ParentChildRetrievalParams(
        parent_collection_suffix="_parent",
    )
    assert sparse_parent_child.sparse == SparseRetrievalParams(
        top_k=5,
        min_score_ratio=0.35,
        gap_ratio=None,
        max_kept=None,
    )
    assert hybrid_parent_child.strategy == "hybrid"
    assert hybrid_parent_child.parent_child == ParentChildRetrievalParams(
        parent_collection_suffix="_parent",
    )
    assert hybrid_parent_child.dense == DenseRetrievalParams(
        top_k=10,
        min_score=0.35,
        max_kept=None,
    )
    assert hybrid_parent_child.sparse == SparseRetrievalParams(
        top_k=10,
        min_score_ratio=0.35,
        gap_ratio=None,
        max_kept=None,
    )
