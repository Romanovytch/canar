from __future__ import annotations

from canar.app.api.embed_client import EmbedClient
from canar.app.config import AppConfig
from canar.app.retrieval.adapters.qdrant import QdrantRetrievalAdapter
from canar.app.retrieval.models import RetrievalHit, RetrievalProfile, RetrievalQuery
from canar.app.retrieval.profiles import AGENT_RETRIEVAL_PROFILES, build_retrieval_profiles
from canar.app.retrieval.strategies.base import RetrievalStrategy
from canar.app.retrieval.strategies.simple_vector import SimpleVectorStrategy


class RetrievalService:
    def __init__(
        self,
        embed_client: EmbedClient,
        profiles: dict[str, RetrievalProfile],
        agent_profiles: dict[str, str | None],
        strategies: dict[str, RetrievalStrategy],
    ):
        self.embed_client = embed_client
        self.profiles = profiles
        self.agent_profiles = agent_profiles
        self.strategies = strategies

    @classmethod
    def from_config(cls, cfg: AppConfig, embed_client: EmbedClient) -> RetrievalService:
        profiles = build_retrieval_profiles(tuple(cfg.qdrant_collections))
        qdrant = QdrantRetrievalAdapter(cfg.qdrant_url, cfg.qdrant_api_key)
        strategies: dict[str, RetrievalStrategy] = {
            "simple_vector": SimpleVectorStrategy(profiles["simple_vector"], qdrant)
        }
        return cls(
            embed_client=embed_client,
            profiles=profiles,
            agent_profiles=AGENT_RETRIEVAL_PROFILES,
            strategies=strategies,
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

        dense_vector = self.embed_client.embed_query(query)
        retrieval_query = RetrievalQuery(
            text=query,
            profile_name=profile.name,
            dense_vector=dense_vector,
        )
        return strategy.search(retrieval_query)
