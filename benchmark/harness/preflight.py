"""
Preflight helpers for the benchmark, kept here so they can be unit-tested
without importing the e2e script (which connects to Qdrant/LLM on import).
"""

from __future__ import annotations

from collections.abc import Callable


def unclassified_profiles(
    profiles,
    needs_dense: Callable[[object], bool],
    needs_sparse: Callable[[object], bool],
) -> list[str]:
    """Profiles that neither predicate claims.

    The benchmark states each profile's vector needs as two lists of names, which
    have to be extended when a profile is added to the product. A profile absent
    from both lists is not one that needs no vectors — every strategy queries at
    least one — it is one nobody has classified yet. Left alone it would be taken
    as needing nothing, and the checks the preflight exists to make would be
    skipped for it in silence, surfacing as a crash mid-run instead.

    This happened: the ``hybrid_summary_rerank_*`` profiles arrived with the
    summary rerank variants and were absent from both lists.
    """
    return [
        profile.name
        for profile in profiles
        if not needs_dense(profile) and not needs_sparse(profile)
    ]


def _blank_requirement() -> dict[str, bool]:
    return {"dense": False, "sparse": False, "source": False, "parent_id": False}


def collection_requirements(
    profiles,
    needs_dense: Callable[[object], bool],
    needs_sparse: Callable[[object], bool],
) -> dict[str, dict[str, bool]]:
    """What each collection the active profiles query must provide.

    Returns {collection_name: {"dense", "sparse", "source", "parent_id"}}. Uses
    the product's own ``RetrievalProfile.effective_collections()``, so a profile
    that derives its collection (e.g. a summary profile, which appends a
    configured suffix) is validated against the derived name automatically — the
    suffix itself is the product's business, not ours. Requirements are unioned
    across profiles that share a collection; the ``source`` payload is only
    required by profiles that apply a source filter.

    A parent-child profile adds two requirements, because its expansion fails
    *silently*: the expander skips any collection whose parent collection is
    missing, and skips any hit without a ``parent_id`` payload, in both cases
    returning the hits untouched. The profile then runs to completion and scores
    exactly like its base profile, which is indistinguishable from "the strategy
    made no difference". So the child collection must carry ``parent_id``, and
    the derived parent collection must exist. The parent collection is only read
    by chunk id, never searched, so it carries no vector requirement.
    """
    reqs: dict[str, dict[str, bool]] = {}
    for profile in profiles:
        for col in profile.effective_collections():
            need = reqs.setdefault(col, _blank_requirement())
            need["dense"] |= needs_dense(profile)
            need["sparse"] |= needs_sparse(profile)
            need["source"] |= profile.source_filter is not None
            if profile.parent_child is None:
                continue
            need["parent_id"] = True
            parent_col = f"{col}{profile.parent_child.parent_collection_suffix}"
            reqs.setdefault(parent_col, _blank_requirement())
    return reqs
