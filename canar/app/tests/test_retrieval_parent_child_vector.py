from __future__ import annotations

from canar.app.retrieval.expanders.parent_child import ParentChildExpander
from canar.app.retrieval.models import (
    ParentChildRetrievalParams,
    RetrievalHit,
)


class FakeAdapter:
    def __init__(self):
        self.existing_collections = {"children_a_parents"}
        self.collection_checks = []
        self.parent_fetches = []

    def collection_exists(self, collection: str) -> bool:
        self.collection_checks.append(collection)
        return collection in self.existing_collections

    def fetch_by_chunk_ids(self, collection: str, chunk_ids: list[str]) -> dict[str, str]:
        self.parent_fetches.append((collection, chunk_ids))
        return {"parent-1": "parent one text", "parent-2": "parent two text"}


def test_parent_child_expander_fetches_parents_and_populates_generation_text():
    adapter = FakeAdapter()
    expander = ParentChildExpander(adapter)
    hits = [
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

    expanded = expander.expand(
        hits,
        ParentChildRetrievalParams(parent_collection_suffix="_parents"),
    )

    assert adapter.collection_checks == ["children_a_parents", "children_b_parents"]
    assert adapter.parent_fetches == [("children_a_parents", ["parent-1", "parent-2"])]
    assert [hit.text for hit in expanded] == ["child one", "child two", "unmapped child"]
    assert [hit.generation_text for hit in expanded] == [
        "parent one text",
        "parent two text",
        "unmapped child",
    ]


def test_parent_child_expander_uses_structured_parent_collection_suffix():
    adapter = FakeAdapter()
    adapter.existing_collections = {"children_a_parents"}
    expander = ParentChildExpander(adapter)
    hits = [
        RetrievalHit(
            text="child one",
            collection="children_a",
            score=0.2,
            score_norm=0,
            metadata={"parent_id": "parent-1"},
        )
    ]

    expander.expand(
        hits,
        ParentChildRetrievalParams(parent_collection_suffix="_parents"),
    )

    assert adapter.collection_checks == ["children_a_parents"]
    assert adapter.parent_fetches == [("children_a_parents", ["parent-1"])]
