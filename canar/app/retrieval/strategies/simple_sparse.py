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

        params = self.profile.sparse
        assert params is not None
        fetch_top_k = (
            params.fetch_top_k if params.fetch_top_k is not None else self.profile.fetch_top_k
        )
        all_hits: list[RetrievalHit] = []
        for collection in self.profile.collections:
            hits = self.adapter.search_sparse(
                collection=collection,
                query_vector=query.sparse_vector,
                top_k=fetch_top_k,
                source_filter=self.profile.source_filter,
                vector_name=params.vector_name,
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
        params = self.profile.sparse
        assert params is not None
        min_score = params.min_score if params.min_score is not None else self.profile.min_score
        pruned = [hit for hit in hits if hit.score_norm >= min_score]
        if params.gap_ratio is not None:
            pruned = self._keep_until_score_gap(pruned, params.gap_ratio)
        if not pruned:
            pruned = hits[: self.profile.fallback_top_k]
        if self.profile.max_results is not None:
            pruned = pruned[: self.profile.max_results]
        return pruned

    def _keep_until_score_gap(
        self,
        hits: list[RetrievalHit],
        gap_ratio: float,
    ) -> list[RetrievalHit]:
        if not hits:
            return []

        kept = [hits[0]]
        for previous, current in zip(hits, hits[1:], strict=False):
            denominator = abs(previous.score_norm) or 1.0
            score_gap = (previous.score_norm - current.score_norm) / denominator
            if score_gap > gap_ratio:
                break
            kept.append(current)
        return kept
