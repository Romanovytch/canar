from __future__ import annotations

from dataclasses import replace

from canar.app.api.embed_client import EmbedClient, FastEmbedClient
from canar.app.config import AppConfig
from canar.app.retrieval.adapters.qdrant import QdrantRetrievalAdapter
from canar.app.retrieval.expanders import HitExpander, ParentChildExpander
from canar.app.retrieval.models import RetrievalHit, RetrievalProfile, RetrievalQuery
from canar.app.retrieval.profiles import AGENT_RETRIEVAL_PROFILES, build_retrieval_profiles
from canar.app.retrieval.rerank.rerankers import Reranker, build_reranker
from canar.app.retrieval.strategies.base import RetrievalStrategy
from canar.app.retrieval.strategies.hybrid import HybridStrategy
from canar.app.retrieval.strategies.simple_sparse import SimpleSparseStrategy
from canar.app.retrieval.strategies.simple_vector import SimpleVectorStrategy


class RetrievalService:
    def __init__(
        self,
        embed_client: EmbedClient,
        profiles: dict[str, RetrievalProfile],
        agent_profiles: dict[str, str | None],
        strategies: dict[str, RetrievalStrategy],
        hit_expanders: dict[str, HitExpander] | None = None,
        sparse_embed_client: FastEmbedClient | None = None,
        reranker: Reranker | None = None,
        rerank_enabled: bool = False,
        rerank_top_n: int | None = None,
    ):
        self.embed_client = embed_client
        self.sparse_embed_client = sparse_embed_client
        self.profiles = profiles
        self.agent_profiles = agent_profiles
        self.strategies = strategies
        self.hit_expanders = hit_expanders or {}
        self.reranker = reranker
        self.rerank_enabled = rerank_enabled
        self.rerank_top_n = rerank_top_n

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        embed_client: EmbedClient,
        sparse_embed_client: FastEmbedClient | None = None,
    ) -> RetrievalService:
        profiles = build_retrieval_profiles(
            tuple(cfg.qdrant_collections),
            dense_vector_name=cfg.qdrant_dense_vector_name,
            sparse_vector_name=cfg.qdrant_sparse_vector_name,
        )
        if sparse_embed_client is None and cfg.fastembed_sparse_model:
            sparse_embed_client = FastEmbedClient(cfg.fastembed_sparse_model)

        reranker = build_reranker(
            cfg.reranker_name,
            device=cfg.rerank_device,
            max_length=cfg.rerank_max_length,
        )

        qdrant = QdrantRetrievalAdapter(cfg.qdrant_url, cfg.qdrant_api_key)
        hybrid_profile = profiles["hybrid"]
        hybrid_dense_params = hybrid_profile.dense_params()
        hybrid_sparse_params = hybrid_profile.sparse_params()
        hybrid_dense_profile = replace(
            hybrid_profile,
            name="hybrid_dense",
            strategy="simple_vector",
            top_k=hybrid_dense_params.top_k,
            dense=hybrid_dense_params,
            vector_name=cfg.qdrant_dense_vector_name or None,
        )
        hybrid_sparse_profile = replace(
            hybrid_profile,
            name="hybrid_sparse",
            strategy="simple_sparse",
            top_k=hybrid_sparse_params.top_k,
            sparse=hybrid_sparse_params,
        )
        simple_vector_strategy = SimpleVectorStrategy(profiles["simple_vector"], qdrant)
        simple_sparse_strategy = SimpleSparseStrategy(profiles["simple_sparse"], qdrant)
        hybrid_strategy = HybridStrategy(
            hybrid_profile,
            SimpleVectorStrategy(hybrid_dense_profile, qdrant),
            SimpleSparseStrategy(hybrid_sparse_profile, qdrant),
        )
        strategies: dict[str, RetrievalStrategy] = {
            "simple_vector": simple_vector_strategy,
            "simple_sparse": simple_sparse_strategy,
            "hybrid": hybrid_strategy,
        }
        return cls(
            embed_client=embed_client,
            profiles=profiles,
            agent_profiles=AGENT_RETRIEVAL_PROFILES,
            strategies=strategies,
            hit_expanders={"parent_child": ParentChildExpander(qdrant)},
            sparse_embed_client=sparse_embed_client,
            reranker=reranker,
            rerank_enabled=cfg.rerank_enabled,
            rerank_top_n=cfg.rerank_top_n,
        )

    def search(
        self,
        agent: str,
        query: str,
        rerank: bool | None = None,
    ) -> list[RetrievalHit]:
        profile_name = self.agent_profiles.get(agent)
        if profile_name is None:
            return []

        profile = self.profiles.get(profile_name)
        if profile is None:
            raise ValueError(f"Unknown retrieval profile for agent {agent!r}: {profile_name!r}")

        strategy = self.strategies.get(profile.strategy)
        if strategy is None:
            raise ValueError(f"Unsupported retrieval strategy: {profile.strategy!r}")

        dense_vector = None
        sparse_vector = None
        if profile.strategy in {"simple_vector", "hybrid"}:
            dense_vector = self.embed_client.embed_query(query)
        if profile.strategy in {"simple_sparse", "hybrid"}:
            if self.sparse_embed_client is None:
                raise ValueError(
                    f"{profile.strategy} retrieval requires FASTEMBED_SPARSE_MODEL "
                    "or an injected sparse embed client"
                )
            sparse_vector = self.sparse_embed_client.embed_query(query)

        retrieval_query = RetrievalQuery(
            text=query,
            profile_name=profile.name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        hits = self._expand_hits(profile, hits)
        should_rerank = self.rerank_enabled if rerank is None else rerank
        if should_rerank and self._supports_rerank(profile):
            if self.reranker is None:
                raise ValueError("rerank=True but no reranker is configured")
            return self.reranker.rerank(
                query=query,
                candidates=hits,
                top_n=self.rerank_top_n,
            )
        return hits

    def _expand_hits(
        self,
        profile: RetrievalProfile,
        hits: list[RetrievalHit],
    ) -> list[RetrievalHit]:
        if profile.parent_child is None:
            return hits

        expander = self.hit_expanders.get("parent_child")
        if expander is None:
            raise ValueError("Unsupported retrieval hit expansion: 'parent_child'")
        return expander.expand(hits, profile.parent_child_params())

    def _supports_rerank(self, profile: RetrievalProfile) -> bool:
        return profile.strategy in {"hybrid", "parent_child_hybrid"}