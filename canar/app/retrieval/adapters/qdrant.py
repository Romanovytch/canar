from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from canar.app.retrieval.models import RetrievalHit, SparseVector


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

        try:
            points = self.client.query_points(**query_args).points
        except Exception as exc:
            vector_hint = f" named {vector_name!r}" if vector_name else ""
            raise RuntimeError(
                f"Dense retrieval failed for collection {collection!r}. "
                f"Ensure the collection contains a compatible dense vector{vector_hint}."
            ) from exc
        return [self._to_hit(collection, point) for point in points]

    def search_sparse(
        self,
        collection: str,
        query_vector: SparseVector,
        top_k: int,
        source_filter: str | None = "utilitr",
        vector_name: str | None = None,
    ) -> list[RetrievalHit]:
        query_args: dict[str, Any] = {
            "collection_name": collection,
            "query": models.SparseVector(
                indices=query_vector.indices,
                values=query_vector.values,
            ),
            "limit": top_k,
            "with_payload": True,
            "with_vectors": False,
            "query_filter": self._source_filter(source_filter),
        }
        if vector_name is not None:
            query_args["using"] = vector_name

        try:
            points = self.client.query_points(**query_args).points
        except Exception as exc:
            vector_hint = f" named {vector_name!r}" if vector_name else ""
            raise RuntimeError(
                f"Sparse retrieval failed for collection {collection!r}. "
                f"Ensure the collection contains a compatible sparse vector{vector_hint}."
            ) from exc
        return [self._to_hit(collection, point) for point in points]

    def fetch_by_chunk_ids(
        self,
        collection: str,
        chunk_ids: list[str],
    ) -> dict[str, str]:
        """
        Return parent chunk text keyed by payload chunk_id.
        """
        if not chunk_ids:
            return {}

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchAny(any=chunk_ids),
                )
            ]
        )
        texts_by_id: dict[str, str] = {}
        next_page_offset = None
        while True:
            points, next_page_offset = self.client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=len(chunk_ids),
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")
                text = payload.get("text")
                if isinstance(chunk_id, str) and isinstance(text, str):
                    texts_by_id[chunk_id] = text
            if next_page_offset is None or len(texts_by_id) >= len(set(chunk_ids)):
                break
        return texts_by_id

    def collection_exists(self, collection: str) -> bool:
        try:
            self.client.get_collection(collection_name=collection)
        except Exception:
            return False
        return True

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
