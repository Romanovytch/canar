from __future__ import annotations

from typing import Any

from canar.app.retrieval.adapters.qdrant import QdrantRetrievalAdapter
from canar.app.retrieval.models import DenseRetrievalParams, RetrievalProfile, RetrievalQuery
from canar.app.retrieval.strategies.simple_vector import SimpleVectorStrategy


def search_qdrant(
    qdrant_url: str,
    api_key: str | None,
    collections: list[str],
    query_vector: list[float],
    top_k_per_collection: int = 5,
    source_filter: str | None = "utilitr",
    vector_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Compatibility wrapper for the previous retrieval API.

    New application code should use canar.app.retrieval.RetrievalService.
    """
    profile = RetrievalProfile(
        name="simple_vector",
        strategy="simple_vector",
        collections=tuple(collections),
        fetch_top_k=top_k_per_collection,
        min_score=0.35,
        source_filter=source_filter,
        fallback_top_k=3,
        dense=DenseRetrievalParams(vector_name=vector_name),
    )
    strategy = SimpleVectorStrategy(profile, QdrantRetrievalAdapter(qdrant_url, api_key))
    hits = strategy.search(
        RetrievalQuery(text="", profile_name=profile.name, dense_vector=query_vector)
    )
    return [
        {
            "collection": hit.collection,
            "score": hit.score,
            "score_norm": hit.score_norm,
            "payload": hit.metadata,
        }
        for hit in hits
    ]
