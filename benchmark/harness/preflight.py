"""
Preflight helpers for the benchmark, kept here so they can be unit-tested
without importing the e2e script (which connects to Qdrant/LLM on import).
"""

from __future__ import annotations

from collections.abc import Callable


def collection_requirements(
    profiles,
    needs_dense: Callable[[object], bool],
    needs_sparse: Callable[[object], bool],
) -> dict[str, dict[str, bool]]:
    """What each collection the active profiles query must provide.

    Returns {collection_name: {"dense", "sparse", "source"}}. Uses the product's
    own ``RetrievalProfile.effective_collections()``, so a summary profile is
    validated against its ``<collection>_summaries`` collection automatically.
    Requirements are unioned across profiles that share a collection; the
    ``source`` payload is only required by profiles that apply a source filter.
    """
    reqs: dict[str, dict[str, bool]] = {}
    for profile in profiles:
        for col in profile.effective_collections():
            need = reqs.setdefault(col, {"dense": False, "sparse": False, "source": False})
            need["dense"] |= needs_dense(profile)
            need["sparse"] |= needs_sparse(profile)
            need["source"] |= profile.source_filter is not None
    return reqs
