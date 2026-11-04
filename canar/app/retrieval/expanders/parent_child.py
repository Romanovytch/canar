from __future__ import annotations

from dataclasses import replace

from canar.app.retrieval.adapters.qdrant import QdrantRetrievalAdapter
from canar.app.retrieval.models import ParentChildRetrievalParams, RetrievalHit


class ParentChildExpander:
    def __init__(self, adapter: QdrantRetrievalAdapter):
        self.adapter = adapter

    def expand(
        self,
        hits: list[RetrievalHit],
        params: ParentChildRetrievalParams,
    ) -> list[RetrievalHit]:
        parent_collections = self._existing_parent_collections(hits, params)
        parent_ids_by_collection = self._parent_ids_by_collection(hits, parent_collections)
        parent_texts = {
            collection: self.adapter.fetch_by_chunk_ids(
                collection=parent_collection,
                chunk_ids=parent_ids,
            )
            for collection, parent_ids in parent_ids_by_collection.items()
            if (parent_collection := parent_collections.get(collection)) is not None
        }

        return [
            replace(
                hit,
                generation_text=self._generation_text(hit, parent_collections, parent_texts),
            )
            for hit in hits
        ]

    def _existing_parent_collections(
        self,
        hits: list[RetrievalHit],
        params: ParentChildRetrievalParams,
    ) -> dict[str, str]:
        collections_with_parent_ids = {
            hit.collection
            for hit in hits
            if isinstance(hit.metadata.get("parent_id"), str) and hit.metadata.get("parent_id")
        }
        parent_collections: dict[str, str] = {}
        for collection in sorted(collections_with_parent_ids):
            parent_collection = f"{collection}{params.parent_collection_suffix}"
            if self.adapter.collection_exists(parent_collection):
                parent_collections[collection] = parent_collection
        return parent_collections

    def _parent_ids_by_collection(
        self,
        hits: list[RetrievalHit],
        parent_collections: dict[str, str],
    ) -> dict[str, list[str]]:
        parent_ids_by_collection: dict[str, list[str]] = {}
        for hit in hits:
            if hit.collection not in parent_collections:
                continue
            parent_id = hit.metadata.get("parent_id")
            if not isinstance(parent_id, str) or not parent_id:
                continue
            ids = parent_ids_by_collection.setdefault(hit.collection, [])
            if parent_id not in ids:
                ids.append(parent_id)
        return parent_ids_by_collection

    def _generation_text(
        self,
        hit: RetrievalHit,
        parent_collections: dict[str, str],
        parent_texts: dict[str, dict[str, str]],
    ) -> str:
        if hit.collection not in parent_collections:
            return hit.text
        parent_id = hit.metadata.get("parent_id")
        if not isinstance(parent_id, str):
            return hit.text
        return parent_texts.get(hit.collection, {}).get(parent_id, hit.text)
