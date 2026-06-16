from __future__ import annotations

from canar.app.api.embed_client import EmbedClient, FastEmbedClient
from canar.app.config import AppConfig
from canar.app.retrieval.adapters.qdrant import QdrantRetrievalAdapter
from canar.app.retrieval.models import RetrievalHit, RetrievalProfile, RetrievalQuery
from canar.app.retrieval.profiles import AGENT_RETRIEVAL_PROFILES, build_retrieval_profiles
from canar.app.retrieval.strategies.base import RetrievalStrategy
from canar.app.retrieval.strategies.simple_sparse import SimpleSparseStrategy
from canar.app.retrieval.strategies.simple_vector import SimpleVectorStrategy


class RetrievalService:
    def __init__(
        self,
        embed_client: EmbedClient,
        profiles: dict[str, RetrievalProfile],
        agent_profiles: dict[str, str | None],
        strategies: dict[str, RetrievalStrategy],
        sparse_embed_client: FastEmbedClient | None = None,
    ):
        self.embed_client = embed_client
        self.sparse_embed_client = sparse_embed_client
        self.profiles = profiles
        self.agent_profiles = agent_profiles
        self.strategies = strategies

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        embed_client: EmbedClient,
        sparse_embed_client: FastEmbedClient | None = None,
    ) -> RetrievalService:
        profiles = build_retrieval_profiles(
            tuple(cfg.qdrant_collections),
            sparse_vector_name=cfg.qdrant_sparse_vector_name,
        )
        if sparse_embed_client is None and cfg.fastembed_sparse_model:
            sparse_embed_client = FastEmbedClient(cfg.fastembed_sparse_model)

        qdrant = QdrantRetrievalAdapter(cfg.qdrant_url, cfg.qdrant_api_key)
        strategies: dict[str, RetrievalStrategy] = {
            "simple_vector": SimpleVectorStrategy(profiles["simple_vector"], qdrant),
            "simple_sparse": SimpleSparseStrategy(profiles["simple_sparse"], qdrant),
        }
        return cls(
            embed_client=embed_client,
            profiles=profiles,
            agent_profiles=AGENT_RETRIEVAL_PROFILES,
            strategies=strategies,
            sparse_embed_client=sparse_embed_client,
        )

    def search(self, agent: str, query: str) -> list[RetrievalHit]:
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
        if profile.strategy == "simple_vector":
            dense_vector = self.embed_client.embed_query(query)
        elif profile.strategy == "simple_sparse":
            if self.sparse_embed_client is None:
                raise ValueError(
                    "simple_sparse retrieval requires FASTEMBED_SPARSE_MODEL "
                    "or an injected sparse embed client"
                )
            sparse_vector = self.sparse_embed_client.embed_query(query)

        retrieval_query = RetrievalQuery(
            text=query,
            profile_name=profile.name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        return strategy.search(retrieval_query)
