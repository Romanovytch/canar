from __future__ import annotations

from dataclasses import replace

from canar.app.retrieval.fusion import FusionStrategy, ReciprocalRankFusion
from canar.app.retrieval.models import RetrievalHit, RetrievalProfile, RetrievalQuery
from canar.app.retrieval.strategies.base import RetrievalStrategy


class HybridStrategy:
    def __init__(
        self,
        profile: RetrievalProfile,
        dense_strategy: RetrievalStrategy,
        sparse_strategy: RetrievalStrategy,
        fusion_strategies: dict[str, FusionStrategy] | None = None,
    ):
        self.profile = profile
        self.dense_strategy = dense_strategy
        self.sparse_strategy = sparse_strategy
        self.fusion_strategies = fusion_strategies or {"rrf": ReciprocalRankFusion()}

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        if query.dense_vector is None:
            raise ValueError("hybrid retrieval requires a dense query vector")
        if query.sparse_vector is None:
            raise ValueError("hybrid retrieval requires a sparse query vector")

        dense_hits = self.dense_strategy.search(
            replace(query, profile_name=f"{self.profile.name}:dense")
        )
        sparse_hits = self.sparse_strategy.search(
            replace(query, profile_name=f"{self.profile.name}:sparse")
        )

        fusion_name = self.profile.fusion or "rrf"
        fusion = self.fusion_strategies.get(fusion_name)
        if fusion is None:
            raise ValueError(f"Unsupported hybrid fusion strategy: {fusion_name!r}")

        return fusion.fuse(
            [dense_hits, sparse_hits],
            top_k=self.profile.final_top_k or self.profile.top_k,
        )
