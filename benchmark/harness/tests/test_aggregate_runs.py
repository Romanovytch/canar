"""Combining runs: what gets averaged, and what gets left out.

The scores a run loses are not the same ones the next run loses, so averaging
whatever is present compares means taken over different question sets. That is
what these tests are mostly about.
"""

from __future__ import annotations

import pandas as pd
from aggregate_runs import aggregate, paired_table, questions_scored_in_every_run


def _rows(records):
    return pd.DataFrame(records)


def test_a_question_scored_everywhere_is_used():
    frame = _rows([
        {"run": "A", "id": "q1", "profile": "p", "faithfulness": 1.0},
        {"run": "B", "id": "q1", "profile": "p", "faithfulness": 0.5},
    ])
    assert questions_scored_in_every_run(frame, "faithfulness") == {"q1"}


def test_a_question_missing_from_one_run_is_dropped():
    """The judge fails on different questions each time; a question it missed
    once cannot contribute to a mean that is meant to be comparable."""
    frame = _rows([
        {"run": "A", "id": "q1", "profile": "p", "faithfulness": 1.0},
        {"run": "A", "id": "q2", "profile": "p", "faithfulness": 0.2},
        {"run": "B", "id": "q1", "profile": "p", "faithfulness": 0.8},
        {"run": "B", "id": "q2", "profile": "p", "faithfulness": None},
    ])
    assert questions_scored_in_every_run(frame, "faithfulness") == {"q1"}


def test_the_mean_ignores_a_question_missing_in_one_run():
    """Without this, run B would average 0.8 alone and run A 0.6, and the
    difference would read as judge noise when it is a change of denominator."""
    frame = _rows([
        {"run": "A", "id": "q1", "profile": "p", "faithfulness": 1.0},
        {"run": "A", "id": "q2", "profile": "p", "faithfulness": 0.2},
        {"run": "B", "id": "q1", "profile": "p", "faithfulness": 1.0},
        {"run": "B", "id": "q2", "profile": "p", "faithfulness": None},
    ])
    row = aggregate(frame, metrics=("faithfulness",)).iloc[0]
    assert row["faithfulness_n"] == 1        # only q1 survives
    assert row["faithfulness_mean"] == 1.0   # not (0.6 + 1.0) / 2
    assert row["faithfulness_sd"] == 0.0


def test_spread_is_reported_across_runs():
    frame = _rows([
        {"run": "A", "id": "q1", "profile": "p", "faithfulness": 1.0},
        {"run": "B", "id": "q1", "profile": "p", "faithfulness": 0.0},
    ])
    row = aggregate(frame, metrics=("faithfulness",)).iloc[0]
    assert row["faithfulness_mean"] == 0.5
    assert row["faithfulness_sd"] == 0.5     # population sd of {1.0, 0.0}


def test_a_deterministic_metric_shows_no_spread():
    """Retrieval metrics reproduce, and a zero spread is the evidence for it."""
    frame = _rows([
        {"run": "A", "id": "q1", "profile": "p", "hit_rate": 1.0},
        {"run": "B", "id": "q1", "profile": "p", "hit_rate": 1.0},
    ])
    row = aggregate(frame, metrics=("hit_rate",)).iloc[0]
    assert row["hit_rate_sd"] == 0.0


def test_profiles_are_kept_apart():
    frame = _rows([
        {"run": "A", "id": "q1", "profile": "dense", "faithfulness": 1.0},
        {"run": "A", "id": "q1", "profile": "hybrid", "faithfulness": 0.0},
    ])
    table = aggregate(frame, metrics=("faithfulness",)).set_index("profile")
    assert table.loc["dense", "faithfulness_mean"] == 1.0
    assert table.loc["hybrid", "faithfulness_mean"] == 0.0


def test_the_paired_table_has_a_row_per_question_and_a_column_per_profile():
    """A paired test compares profiles question by question, not in aggregate."""
    frame = _rows([
        {"run": "A", "id": "q1", "profile": "dense", "faithfulness": 1.0},
        {"run": "A", "id": "q1", "profile": "hybrid", "faithfulness": 0.5},
        {"run": "B", "id": "q1", "profile": "dense", "faithfulness": 0.8},
        {"run": "B", "id": "q1", "profile": "hybrid", "faithfulness": 0.5},
    ])
    table = paired_table(frame, "faithfulness")
    assert list(table.index) == ["q1"]
    assert sorted(table.columns) == ["dense", "hybrid"]
    assert table.loc["q1", "dense"] == 0.9   # averaged over the two runs
