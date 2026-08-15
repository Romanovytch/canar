"""
Preflight helpers for the benchmark, kept here so they can be unit-tested
without importing the e2e script (which connects to Qdrant/LLM on import).
"""

from __future__ import annotations

from collections.abc import Callable

# Which query vectors a profile needs, read from its strategy rather than from a
# list of profile names. RetrievalService.search decides the same way, so the two
# cannot drift: a profile added to the product is classified here the moment it
# exists, with no benchmark-side edit. A name list had to be extended by hand for
# every new profile, and silently classified the ones it did not know as needing
# nothing — which skipped their preflight checks.
DENSE_STRATEGIES = frozenset({"simple_vector", "hybrid"})
SPARSE_STRATEGIES = frozenset({"simple_sparse", "hybrid"})


def profile_needs_dense(profile) -> bool:
    """Whether the profile issues a dense query and so needs a dense vector."""
    return profile.strategy in DENSE_STRATEGIES


def profile_needs_sparse(profile) -> bool:
    """Whether the profile issues a sparse query and so needs a sparse vector."""
    return profile.strategy in SPARSE_STRATEGIES


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
