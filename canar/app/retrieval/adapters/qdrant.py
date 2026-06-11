from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from canar.app.retrieval.models import RetrievalHit


class QdrantRetrievalAdapter:
    def __init__(self, url: str, api_key: str | None = None):
        self.client = QdrantClient(url=url, api_key=api_key or None)

    def search_dense(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        source_filter: str | None = "utilitr",
        vector_name: str | None = None,
    ) -> list[RetrievalHit]:
        query_args: dict[str, Any] = {
            "collection_name": collection,
            "query": query_vector,
            "limit": top_k,
            "with_payload": True,
            "with_vectors": False,
            "query_filter": self._source_filter(source_filter),
        }
        if vector_name is not None:
            query_args["using"] = vector_name

        points = self.client.query_points(**query_args).points
        return [self._to_hit(collection, point) for point in points]

    def _source_filter(self, source_filter: str | None) -> models.Filter | None:
        if not source_filter:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source_filter))]
        )

    def _to_hit(self, collection: str, point: Any) -> RetrievalHit:
        payload = point.payload or {}
        return RetrievalHit(
            text=payload.get("text") or "",
            collection=collection,
            score=point.score,
            score_norm=0.0,
            source=payload.get("source"),
            source_url=payload.get("url") or payload.get("source_url"),
            section=payload.get("section") or "",
            metadata=dict(payload),
        )
