from __future__ import annotations

from canar.app.retrieval.models import (
    ParentChildRetrievalParams,
    RetrievalHit,
    RetrievalProfile,
    RetrievalQuery,
)
from canar.app.retrieval.strategies.parent_child import ParentChildStrategy


class FakeAdapter:
    def __init__(self):
        self.existing_collections = {"children_a_parent"}
        self.collection_checks = []
        self.parent_fetches = []

    def collection_exists(self, collection: str) -> bool:
        self.collection_checks.append(collection)
        return collection in self.existing_collections

    def fetch_by_chunk_ids(self, collection: str, chunk_ids: list[str]) -> dict[str, str]:
        self.parent_fetches.append((collection, chunk_ids))
        return {"parent-1": "parent one text", "parent-2": "parent two text"}


class FakeChildStrategy:
    def __init__(self):
        self.queries: list[RetrievalQuery] = []

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        self.queries.append(query)
        return [
            RetrievalHit(
                text="child one",
                collection="children_a",
                score=0.2,
                score_norm=0,
                metadata={"parent_id": "parent-1"},
            ),
            RetrievalHit(
                text="child two",
                collection="children_a",
                score=0.8,
                score_norm=1.0,
                metadata={"parent_id": "parent-2"},
            ),
            RetrievalHit(
                text="unmapped child",
                collection="children_b",
                score=0.5,
                score_norm=0.5,
                metadata={"parent_id": "parent-3"},
            ),
        ]


def test_parent_child_vector_fetches_parents_and_populates_generation_text():
    adapter = FakeAdapter()
    child_strategy = FakeChildStrategy()
    profile = RetrievalProfile(
        name="parent_child_vector",
        strategy="parent_child_vector",
        collections=("children_a", "children_b"),
        score_threshold=0.0,
    )
    strategy = ParentChildStrategy(profile, child_strategy, adapter)

    query = RetrievalQuery(
        text="question",
        profile_name="parent_child_vector",
        dense_vector=[0.1, 0.2],
    )
    hits = strategy.search(query)

    assert child_strategy.queries == [query]
    assert adapter.collection_checks == ["children_a_parent", "children_b_parent"]
    assert adapter.parent_fetches == [("children_a_parent", ["parent-1", "parent-2"])]
    assert [hit.text for hit in hits] == ["child one", "child two", "unmapped child"]
    assert [hit.generation_text for hit in hits] == [
        "parent one text",
        "parent two text",
        "unmapped child",
    ]


def test_parent_child_uses_structured_parent_collection_suffix():
    adapter = FakeAdapter()
    adapter.existing_collections = {"children_a_parents"}
    child_strategy = FakeChildStrategy()
    profile = RetrievalProfile(
        name="parent_child_vector",
        strategy="parent_child_vector",
        collections=("children_a", "children_b"),
        parent_child=ParentChildRetrievalParams(parent_collection_suffix="_parents"),
    )
    strategy = ParentChildStrategy(profile, child_strategy, adapter)

    strategy.search(
        RetrievalQuery(
            text="question",
            profile_name="parent_child_vector",
            dense_vector=[0.1],
        )
    )

    assert adapter.collection_checks == ["children_a_parents", "children_b_parents"]
    assert adapter.parent_fetches == [("children_a_parents", ["parent-1", "parent-2"])]


def test_parent_child_legacy_suffix_resolves_for_backward_compatibility():
    profile = RetrievalProfile(
        name="legacy_parent_child",
        strategy="parent_child_vector",
        collections=("children_a",),
        parent_collection_suffix="_parents",
    )

    assert profile.parent_child_params() == ParentChildRetrievalParams(
        parent_collection_suffix="_parents",
    )


def test_parent_child_structured_params_win_over_legacy_suffix():
    profile = RetrievalProfile(
        name="parent_child",
        strategy="parent_child_vector",
        collections=("children_a",),
        parent_child=ParentChildRetrievalParams(parent_collection_suffix="_structured"),
        parent_collection_suffix="_legacy",
    )

    assert profile.parent_child_params() == ParentChildRetrievalParams(
        parent_collection_suffix="_structured",
    )
