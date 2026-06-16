from __future__ import annotations

from dataclasses import replace

from canar.app.retrieval.adapters.qdrant import QdrantRetrievalAdapter
from canar.app.retrieval.models import RetrievalHit, RetrievalProfile, RetrievalQuery


class SimpleSparseStrategy:
    def __init__(self, profile: RetrievalProfile, adapter: QdrantRetrievalAdapter):
        self.profile = profile
        self.adapter = adapter

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        if query.sparse_vector is None:
            raise ValueError("simple_sparse retrieval requires a sparse query vector")

        all_hits: list[RetrievalHit] = []
        for collection in self.profile.collections:
            hits = self.adapter.search_sparse(
                collection=collection,
                query_vector=query.sparse_vector,
                top_k=self.profile.top_k,
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
        pruned = [hit for hit in hits if hit.score_norm >= self.profile.score_threshold]
        return pruned or hits[: self.profile.fallback_top_k]
