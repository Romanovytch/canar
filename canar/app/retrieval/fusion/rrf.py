from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import replace
from math import isfinite
from numbers import Real

from canar.app.retrieval.models import RetrievalHit


class ReciprocalRankFusion:
    def __init__(self, rank_constant: int = 60):
        self.rank_constant = rank_constant

    def fuse(
        self,
        ranked_lists: Sequence[Sequence[RetrievalHit]],
        top_k: int,
        weights: Sequence[float] | None = None,
    ) -> list[RetrievalHit]:
        normalized_weights = self._validate_weights(ranked_lists, weights)
        scores: dict[Hashable, float] = {}
        representatives: dict[Hashable, RetrievalHit] = {}

        for ranked_list, weight in zip(ranked_lists, normalized_weights, strict=True):
            for rank, hit in enumerate(ranked_list, start=1):
                key = self._hit_key(hit)
                scores[key] = scores.get(key, 0.0) + weight / (self.rank_constant + rank)
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

    def _validate_weights(
        self,
        ranked_lists: Sequence[Sequence[RetrievalHit]],
        weights: Sequence[float] | None,
    ) -> list[float]:
        if weights is None:
            return [1.0] * len(ranked_lists)

        if len(weights) != len(ranked_lists):
            raise ValueError(
                "RRF weights must match the number of ranked lists "
                f"({len(weights)} weights for {len(ranked_lists)} lists)"
            )

        normalized_weights: list[float] = []
        for index, weight in enumerate(weights):
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise ValueError(f"RRF weight at index {index} must be numeric")
            weight_float = float(weight)
            if not isfinite(weight_float):
                raise ValueError(f"RRF weight at index {index} must be finite")
            if weight_float < 0.0:
                raise ValueError(f"RRF weight at index {index} must be non-negative")
            normalized_weights.append(weight_float)

        if normalized_weights and all(weight == 0.0 for weight in normalized_weights):
            raise ValueError("At least one RRF weight must be greater than zero")
        return normalized_weights

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
