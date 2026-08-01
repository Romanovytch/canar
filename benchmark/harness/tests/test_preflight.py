"""Preflight: which collections each profile makes the benchmark validate.

A summary profile queries a derived collection (the base name plus the suffix
the product configures), so the preflight must validate that one too, not just
the base collection. The suffix used below is only an example."""

from dataclasses import dataclass

from preflight import collection_requirements


@dataclass
class _FakeProfile:
    """Duck-types the bits of RetrievalProfile the helper reads."""
    name: str
    collections: tuple
    source_filter: str | None = None
    summary_suffix: str | None = None

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
    assert reqs == {"utilitr_v2": {"dense": True, "sparse": True, "source": True}}


def test_summary_profile_validates_the_summaries_collection():
    profiles = [_FakeProfile("hybrid_summary", ("utilitr_v2",), summary_suffix="_summaries")]
    reqs = collection_requirements(profiles, _needs_dense, _needs_sparse)

    # it points at the derived collection, not the base one
    assert "utilitr_v2_summaries" in reqs
    assert "utilitr_v2" not in reqs
    # summary retrieval is hybrid -> needs dense + sparse; no source filter here
    assert reqs["utilitr_v2_summaries"] == {"dense": True, "sparse": True, "source": False}


def test_requirements_are_unioned_across_profiles():
    # simple_vector -> dense + source; simple_sparse -> sparse, no source
    profiles = [
        _FakeProfile("simple_vector", ("utilitr_v2",), source_filter="utilitr"),
        _FakeProfile("simple_sparse", ("utilitr_v2",)),
    ]
    reqs = collection_requirements(profiles, _needs_dense, _needs_sparse)
    assert reqs["utilitr_v2"] == {"dense": True, "sparse": True, "source": True}
