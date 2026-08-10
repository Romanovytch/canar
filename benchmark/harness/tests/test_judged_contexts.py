"""The benchmark must judge the answer against the text the model actually saw.

``build_messages`` builds the prompt from ``generation_text or text``, so a
parent-child profile generates from the expanded parent passage while ``text``
still holds the child chunk that matched the query. Handing ``text`` to RAGAS
scores faithfulness against a text the model never read, and the bias grows with
how much the expansion contributed — the better the strategy works, the worse it
scores. Profiles without expansion are unaffected, so this shifts the comparison
between strategies, not just the absolute numbers.

The pipeline cannot be imported here (``eval_e2e.py`` builds Qdrant and LLM
clients at import time), so the contract is covered against a real expander and
the call site is then guarded by reading its source.
"""

from __future__ import annotations

from pathlib import Path

from canar.app.agents import generic_agent
from canar.app.retrieval.expanders.parent_child import ParentChildExpander
from canar.app.retrieval.models import ParentChildRetrievalParams, RetrievalHit

EVAL_E2E = Path(__file__).resolve().parents[2] / "e2e" / "eval_e2e.py"

CHILD = "open_dataset is the recommended function."
PARENT = (
    "Reading Parquet in R. "
    + CHILD
    + " read_parquet loads everything into memory, which is why open_dataset is "
    "preferred for large files."
)


class _StubAdapter:
    """Stands in for QdrantRetrievalAdapter: the parent collection exists and
    holds one parent passage."""

    def __init__(self, existing: set[str]):
        self.existing = existing

    def collection_exists(self, name: str) -> bool:
        return name in self.existing

    def fetch_by_chunk_ids(self, collection: str, chunk_ids: list[str]) -> dict[str, str]:
        return {"p1": PARENT}


def _hits() -> list[RetrievalHit]:
    return [
        RetrievalHit(
            text=CHILD,
            collection="col",
            score=0.9,
            score_norm=1.0,
            section="Parquet",
            metadata={"parent_id": "p1"},
        )
    ]


def _prompt_text(hits: list[RetrievalHit]) -> str:
    messages, _ = generic_agent.build_messages("Which function should I use?", hits)
    return "\n".join(message["content"] for message in messages)


def _judged_contexts(hits: list[RetrievalHit]) -> list[str]:
    """The expression eval_e2e.py fills PipelineOutput.contexts with."""
    return [hit.generation_text or hit.text for hit in hits]


def test_expanded_hits_are_judged_against_the_parent_passage():
    expander = ParentChildExpander(_StubAdapter({"col_parent"}))
    hits = expander.expand(_hits(), ParentChildRetrievalParams())

    # the prompt carries the parent passage, so the judged context must too
    assert PARENT in _prompt_text(hits)
    assert _judged_contexts(hits) == [PARENT]
    # the child chunk alone would be the pre-fix behavior
    assert [hit.text for hit in hits] == [CHILD]


def test_hits_without_expansion_are_judged_against_their_own_text():
    hits = _hits()  # no expander ran, so generation_text is unset

    assert _judged_contexts(hits) == [CHILD]
    assert _judged_contexts(hits) == [hit.text for hit in hits]


def test_expansion_is_skipped_when_the_parent_collection_is_absent():
    """The expander fails silently, which is what the preflight now guards."""
    expander = ParentChildExpander(_StubAdapter(set()))
    hits = expander.expand(_hits(), ParentChildRetrievalParams())

    assert _judged_contexts(hits) == [CHILD]


def test_eval_e2e_fills_contexts_from_generation_text():
    """Guards the call site, not just the contract.

    The tests above pass whether or not eval_e2e.py applies the rule, so they
    cannot catch the line being reverted to ``hit.text`` — which breaks nothing
    visibly, since the benchmark still runs and still reports numbers. Checking
    the source is blunt, but it is what actually fails when a merge drops it.
    """
    source = EVAL_E2E.read_text(encoding="utf-8")

    assert "contexts=[hit.generation_text or hit.text for hit in citations]" in source, (
        "eval_e2e.py must fill PipelineOutput.contexts from `generation_text or text`, "
        "the same expression build_messages uses to build the prompt. If the line was "
        "only reworded, update this test; if it was reverted to `hit.text`, restore it."
    )
