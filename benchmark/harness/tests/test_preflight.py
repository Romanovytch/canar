"""Preflight: which collections each profile makes the benchmark validate.

A summary profile queries a derived collection (the base name plus the suffix
the product configures), so the preflight must validate that one too, not just
the base collection. The suffix used below is only an example."""

from dataclasses import dataclass

from preflight import collection_requirements, unclassified_profiles


@dataclass
class _FakeParentChild:
    """Duck-types ParentChildRetrievalParams."""
    parent_collection_suffix: str = "_parent"


@dataclass
class _FakeProfile:
    """Duck-types the bits of RetrievalProfile the helper reads."""
    name: str
    collections: tuple
    source_filter: str | None = None
    summary_suffix: str | None = None
    parent_child: _FakeParentChild | None = None

    def effective_collections(self):
        if self.summary_suffix is None:
            return self.collections
        return tuple(f"{c}{self.summary_suffix}" for c in self.collections)


def _needs_dense(profile):
    return profile.name in {"hybrid", "hybrid_summary", "simple_vector"}


def _needs_sparse(profile):
    return profile.name in {"hybrid", "hybrid_summary", "simple_sparse"}


def test_base_collection_requirements():
    profiles = [_FakeProfile("hybrid", ("utilitr_v2",), source_filter="utilitr")]
    reqs = collection_requirements(profiles, _needs_dense, _needs_sparse)
    assert reqs == {
        "utilitr_v2": {"dense": True, "sparse": True, "source": True, "parent_id": False}
    }


def test_summary_profile_validates_the_summaries_collection():
    profiles = [_FakeProfile("hybrid_summary", ("utilitr_v2",), summary_suffix="_summaries")]
    reqs = collection_requirements(profiles, _needs_dense, _needs_sparse)

    # it points at the derived collection, not the base one
    assert "utilitr_v2_summaries" in reqs
    assert "utilitr_v2" not in reqs
    # summary retrieval is hybrid -> needs dense + sparse; no source filter here
    assert reqs["utilitr_v2_summaries"] == {
        "dense": True, "sparse": True, "source": False, "parent_id": False
    }


def test_requirements_are_unioned_across_profiles():
    # simple_vector -> dense + source; simple_sparse -> sparse, no source
    profiles = [
        _FakeProfile("simple_vector", ("utilitr_v2",), source_filter="utilitr"),
        _FakeProfile("simple_sparse", ("utilitr_v2",)),
    ]
    reqs = collection_requirements(profiles, _needs_dense, _needs_sparse)
    assert reqs["utilitr_v2"] == {
        "dense": True, "sparse": True, "source": True, "parent_id": False
    }


def test_parent_child_profile_validates_the_parent_collection():
    profiles = [
        _FakeProfile("hybrid", ("utilitr_v2",), parent_child=_FakeParentChild()),
    ]
    reqs = collection_requirements(profiles, _needs_dense, _needs_sparse)

    # the child collection must carry parent_id, or the expansion finds nothing
    assert reqs["utilitr_v2"]["parent_id"] is True
    # the parent collection must exist, but it is fetched by chunk id and never
    # searched, so it needs no vector
    assert reqs["utilitr_v2_parent"] == {
        "dense": False, "sparse": False, "source": False, "parent_id": False
    }


def test_parent_collection_derives_from_the_effective_collection():
    """A profile that both derives its collection and expands to a parent must
    validate the parent of the *derived* name, not of the base one."""
    profiles = [
        _FakeProfile(
            "hybrid",
            ("utilitr_v2",),
            summary_suffix="_summaries",
            parent_child=_FakeParentChild(),
        ),
    ]
    reqs = collection_requirements(profiles, _needs_dense, _needs_sparse)

    assert "utilitr_v2_summaries_parent" in reqs
    assert "utilitr_v2_parent" not in reqs


def test_profiles_without_parent_child_require_no_parent_collection():
    profiles = [_FakeProfile("hybrid", ("utilitr_v2",))]
    reqs = collection_requirements(profiles, _needs_dense, _needs_sparse)
    assert list(reqs) == ["utilitr_v2"]


def test_a_profile_in_both_lists_is_classified():
    profiles = [_FakeProfile("hybrid", ("utilitr_v2",))]
    assert unclassified_profiles(profiles, _needs_dense, _needs_sparse) == []


def test_a_profile_in_one_list_is_classified():
    """Dense-only and sparse-only profiles are complete answers, not gaps."""
    profiles = [
        _FakeProfile("simple_vector", ("utilitr_v2",)),
        _FakeProfile("simple_sparse", ("utilitr_v2",)),
    ]
    assert unclassified_profiles(profiles, _needs_dense, _needs_sparse) == []


def test_a_profile_in_neither_list_is_reported():
    """The gap this exists to catch.

    A profile added to the product but not to the name lists reads as needing no
    vectors, so its collection checks are skipped in silence and the run fails
    later instead. It happened with the hybrid_summary_rerank_* profiles.
    """
    profiles = [
        _FakeProfile("hybrid", ("utilitr_v2",)),
        _FakeProfile("hybrid_summary_rerank_bge", ("utilitr_v2",)),
    ]
    assert unclassified_profiles(profiles, _needs_dense, _needs_sparse) == [
        "hybrid_summary_rerank_bge"
    ]
