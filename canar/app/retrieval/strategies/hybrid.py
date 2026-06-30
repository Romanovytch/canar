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
        fusion_params = profile.fusion_params()
        rrf = ReciprocalRankFusion(rank_constant=fusion_params.rrf_k)
        self.fusion_strategies = fusion_strategies or {
            "rrf": rrf,
            "weighted_rrf": rrf,
        }

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

        fusion_params = self.profile.fusion_params()
        fusion = self.fusion_strategies.get(fusion_params.method)
        if fusion is None:
            raise ValueError(f"Unsupported hybrid fusion strategy: {fusion_params.method!r}")

        return fusion.fuse(
            # Weight order follows the ranked-list order, matching Qdrant's prefetch semantics.
            [dense_hits, sparse_hits],
            top_k=fusion_params.final_top_k,
            weights=[
                fusion_params.weights.get("dense", 1.0),
                fusion_params.weights.get("sparse", 1.0),
            ],
        )
