from __future__ import annotations
from typing import Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


def search_qdrant(qdrant_url: str, api_key: str | None, collections: list[str],
                  query_vector: list[float], top_k_per_collection: int = 5,
                  source_filter: str | None = "utilitr") -> list[Dict[str, Any]]:
    """
    Returns a unified list of hits across collections with normalized per-collection score.
    """
    client = QdrantClient(url=qdrant_url, api_key=api_key or None)
    all_hits = []
    for col in collections:
        flt = None
        if source_filter:
            flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_filter))])
        hits = client.search(collection_name=col, query_vector=query_vector,
                             limit=top_k_per_collection,
                             with_payload=True, with_vectors=False, query_filter=flt)
        if not hits:
            continue
        # min-max normalize within the collection to make cross-collection fusion saner
        scores = [h.score for h in hits]
        lo, hi = min(scores), max(scores)
        rng = (hi - lo) or 1.0
        for h in hits:
            norm = (h.score - lo) / rng
            all_hits.append({
                "collection": col,
                "score": h.score,
                "score_norm": norm,
                "payload": h.payload
            })
    # sort by normalized score then raw score
    all_hits.sort(key=lambda x: (x["score_norm"], x["score"]), reverse=True)
    # filter out weak tails (often irrelevant): keep those with score_norm >= 0.35 or the top-3
    pruned = [h for h in all_hits if h["score_norm"] >= 0.35] or all_hits[:3]
    return pruned
