from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import replace

from canar.app.retrieval.models import RetrievalHit


class ReciprocalRankFusion:
    def __init__(self, rank_constant: int = 60):
        self.rank_constant = rank_constant

    def fuse(
        self,
        ranked_lists: Sequence[Sequence[RetrievalHit]],
        top_k: int,
    ) -> list[RetrievalHit]:
        scores: dict[Hashable, float] = {}
        representatives: dict[Hashable, RetrievalHit] = {}

        for ranked_list in ranked_lists:
            for rank, hit in enumerate(ranked_list, start=1):
                key = self._hit_key(hit)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rank_constant + rank)
                representatives.setdefault(key, hit)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ordered:
            return []

        max_score = ordered[0][1] or 1.0
        fused_hits: list[RetrievalHit] = []
        for key, score in ordered[:top_k]:
            fused_hits.append(
                replace(
                    representatives[key],
                    score=score,
                    score_norm=score / max_score,
                )
            )
        return fused_hits

    def _hit_key(self, hit: RetrievalHit) -> Hashable:
        metadata_id = self._metadata_id(hit)
        if metadata_id is not None:
            return ("metadata_id", hit.collection, metadata_id)
        if hit.source_url and hit.section:
            return ("source_section", hit.collection, hit.source_url, hit.section)
        if hit.source_url:
            return ("source_url_text", hit.collection, hit.source_url, hit.text)
        return ("text", hit.collection, hit.text)

    def _metadata_id(self, hit: RetrievalHit) -> str | None:
        for key in ("id", "_id", "point_id", "chunk_id", "document_id", "doc_id"):
            value = hit.metadata.get(key)
            if value is not None:
                return str(value)
        return None
