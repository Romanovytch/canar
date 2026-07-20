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
        rerankers: dict[str, Reranker] | None = None,
    ):
        self.embed_client = embed_client
        self.sparse_embed_client = sparse_embed_client
        self.profiles = profiles
        self.agent_profiles = agent_profiles
        self.strategies = strategies
        self.hit_expanders = hit_expanders or {}
        self.rerankers = rerankers or {}
        self.reranker = reranker or next(iter(self.rerankers.values()), None)

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        embed_client: EmbedClient,
        sparse_embed_client: FastEmbedClient | None = None,
        profiles: dict[str, RetrievalProfile] | None = None,
        agent_profiles: dict[str, str | None] | None = None,
    ) -> RetrievalService:
        if profiles is None:
            profiles = build_retrieval_profiles(
                tuple(cfg.qdrant_collections),
                dense_vector_name=cfg.qdrant_dense_vector_name,
                sparse_vector_name=cfg.qdrant_sparse_vector_name,
            )
        if agent_profiles is None:
            agent_profiles = AGENT_RETRIEVAL_PROFILES
        if sparse_embed_client is None and cfg.fastembed_sparse_model:
            sparse_embed_client = FastEmbedClient(cfg.fastembed_sparse_model)

        active_profile_names = list(
            dict.fromkeys(name for name in agent_profiles.values() if name is not None)
        )
        rerankers: dict[str, Reranker] = {}
        reranker_cache: dict[tuple[str, str | None, int], Reranker] = {}
        for profile_name in active_profile_names:
            profile = profiles.get(profile_name)
            if profile is None:
                continue
            params = profile.rerank
            if params is None:
                continue
            cache_key = (params.model, params.device, params.max_length)
            reranker = reranker_cache.get(cache_key)
            if reranker is None:
                reranker = build_reranker(
                    params.model,
                    device=params.device,
                    max_length=params.max_length,
                )
                reranker_cache[cache_key] = reranker
            rerankers[profile_name] = reranker

        qdrant = QdrantRetrievalAdapter(cfg.qdrant_url, cfg.qdrant_api_key)

        def build_hybrid_strategy(profile: RetrievalProfile) -> HybridStrategy:
            hybrid_dense_profile = replace(
                profile,
                name=f"{profile.name}:dense",
                strategy="simple_vector",
                dense=profile.dense,
                sparse=None,
                fusion=None,
                rerank=None,
                parent_child=None,
            )
            hybrid_sparse_profile = replace(
                profile,
                name=f"{profile.name}:sparse",
                strategy="simple_sparse",
                dense=None,
                sparse=profile.sparse,
                fusion=None,
                rerank=None,
                parent_child=None,
            )
            return HybridStrategy(
                profile,
                SimpleVectorStrategy(hybrid_dense_profile, qdrant),
                SimpleSparseStrategy(hybrid_sparse_profile, qdrant),
            )

        strategies: dict[str, RetrievalStrategy] = {}
        for profile in profiles.values():
            if profile.strategy == "simple_vector":
                strategies[profile.name] = SimpleVectorStrategy(profile, qdrant)
            elif profile.strategy == "simple_sparse":
                strategies[profile.name] = SimpleSparseStrategy(profile, qdrant)
            elif profile.strategy == "hybrid":
                strategies[profile.name] = build_hybrid_strategy(profile)

        return cls(
            embed_client=embed_client,
            profiles=profiles,
            agent_profiles=agent_profiles,
            strategies=strategies,
            hit_expanders={"parent_child": ParentChildExpander(qdrant)},
            sparse_embed_client=sparse_embed_client,
            rerankers=rerankers,
        )

    def search(
        self,
        agent: str,
        query: str,
    ) -> list[RetrievalHit]:
        profile_name = self.agent_profiles.get(agent)
        if profile_name is None:
            return []

        profile = self.profiles.get(profile_name)
        if profile is None:
            raise ValueError(f"Unknown retrieval profile for agent {agent!r}: {profile_name!r}")

        strategy = self.strategies.get(profile.name)
        if strategy is None:
            raise ValueError(f"Unsupported retrieval profile: {profile.name!r}")

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

        rerank_params = profile.rerank
        should_rerank = rerank_params is not None
        retrieval_query = RetrievalQuery(
            text=query,
            profile_name=profile.name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        hits = strategy.search(retrieval_query)
        hits = self._expand_hits(profile, hits)
        if should_rerank:
            reranker = self.rerankers.get(profile.name) or self.reranker
            if reranker is None:
                raise ValueError("Rerank is enabled but no reranker is configured")
            return reranker.rerank(
                query=query,
                candidates=hits,
                top_k=(
                    rerank_params.output_top_k
                    if rerank_params.output_top_k is not None
                    else profile.output_top_k
                ),
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
        return expander.expand(hits, profile.parent_child)
