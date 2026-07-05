from __future__ import annotations

from dataclasses import replace

from canar.app.retrieval.adapters.qdrant import QdrantRetrievalAdapter
from canar.app.retrieval.models import RetrievalHit, RetrievalProfile, RetrievalQuery


class SimpleVectorStrategy:
    def __init__(self, profile: RetrievalProfile, adapter: QdrantRetrievalAdapter):
        self.profile = profile
        self.adapter = adapter

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        if query.dense_vector is None:
            raise ValueError("simple_vector retrieval requires a dense query vector")

        params = self.profile.dense_params()
        all_hits: list[RetrievalHit] = []
        for collection in self.profile.collections:
            hits = self.adapter.search_dense(
                collection=collection,
                query_vector=query.dense_vector,
                top_k=params.top_k,
                source_filter=self.profile.source_filter,
                vector_name=self.profile.vector_name,
            )
            all_hits.extend(self._normalize_collection_scores(hits))

        all_hits.sort(key=lambda hit: (hit.score_norm, hit.score), reverse=True)
        return self._prune_hits(all_hits)

    def _normalize_collection_scores(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return []

        scores = [hit.score for hit in hits]
        lo, hi = min(scores), max(scores)
        score_range = (hi - lo) or 1.0
        return [replace(hit, score_norm=(hit.score - lo) / score_range) for hit in hits]

    def _prune_hits(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        params = self.profile.dense_params()
        pruned = [
            hit for hit in hits if params.min_score is None or hit.score_norm >= params.min_score
        ]
        if not pruned:
            pruned = hits[: self.profile.fallback_top_k]
        if params.max_kept is not None:
            pruned = pruned[: params.max_kept]
        return pruned
