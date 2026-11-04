"""Judging each question several times: what the score and its spread are.

The judge is an LLM and does not always produce a usable score, so most of
these tests are about what happens to a question when only some of its draws
work, and about the spread landing on the right row.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ragas")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "e2e"))

import judge_repeats  # noqa: E402
from judge_repeats import mean_of_draws, spread_columns, write_spread  # noqa: E402


def _judge(outcomes):
    """A judge whose draws return, or raise, `outcomes` in order."""
    remaining = list(outcomes)

    async def score(sample, callbacks):
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return score


def _run(outcomes, repeats, report=None):
    return asyncio.run(mean_of_draws(_judge(outcomes), None, None, repeats, report))


@pytest.fixture(autouse=True)
def _empty_draws():
    judge_repeats.DRAWS.clear()
    yield
    judge_repeats.DRAWS.clear()


def test_the_score_is_the_mean_of_the_draws():
    assert _run([0.2, 0.4, 0.6], 3) == pytest.approx(0.4)


def test_every_draw_is_requested():
    """N repeats must cost N judgements, not stop at the first success."""
    calls = []

    async def score(sample, callbacks):
        calls.append(1)
        return 0.5

    asyncio.run(mean_of_draws(score, None, None, 5))
    assert len(calls) == 5


def test_one_repeat_returns_the_single_draw_unchanged():
    """judge_repeats=1 has to reproduce the un-repeated benchmark."""
    assert _run([0.37], 1) == pytest.approx(0.37)


def test_an_unscorable_draw_does_not_discard_the_question():
    """RAGAS returns NaN when it cannot score; a plain mean would lose the
    question and the two real scores with it."""
    assert _run([0.5, float("nan"), 0.7], 3) == pytest.approx(0.6)


def test_a_draw_that_raises_does_not_discard_the_question():
    """RAGAS raises when the judge's reply does not parse or a server call
    fails, rather than returning NaN."""
    assert _run([0.6, RuntimeError("unparseable"), 0.8], 3) == pytest.approx(0.7)


def test_a_question_whose_every_draw_fails_fails_as_it_would_unrepeated():
    with pytest.raises(RuntimeError):
        _run([RuntimeError("boom"), RuntimeError("boom")], 2)
    assert np.isnan(_run([float("nan"), float("nan")], 2))


def test_the_report_sees_the_usable_draws_and_the_failures():
    seen = []
    _run([0.4, RuntimeError("500"), 0.6], 3, report=lambda *a: seen.append(a))
    valid, failed, error = seen[0]
    assert valid == [0.4, 0.6] and failed == 1 and isinstance(error, RuntimeError)


def _results(*rows):
    return pd.DataFrame(rows, columns=["user_input", "response", "pipeline_status"])


def _record(metric, question, answer, draws):
    judge_repeats.DRAWS[(metric, question, answer)] = draws


def test_spread_lands_on_the_row_it_belongs_to():
    _record("faithfulness", "q1", "a1", [1.0, 0.5, 1.0])
    _record("faithfulness", "q2", "a2", [0.8, 0.8])
    spread = spread_columns(_results(("q2", "a2", "OK"), ("q1", "a1", "OK")))
    assert spread["faithfulness_sd"].tolist() == pytest.approx([0.0, np.std([1.0, 0.5, 1.0])])
    assert spread["faithfulness_draws"].tolist() == [2, 3]


def test_the_same_question_answered_differently_keeps_its_own_spread():
    """The key includes the answer: two profiles can answer one question
    differently, and each answer has its own judgements."""
    _record("answer_relevancy", "q1", "short", [0.9, 0.9])
    _record("answer_relevancy", "q1", "long", [0.2, 0.8])
    spread = spread_columns(_results(("q1", "long", "OK")))
    assert spread["answer_relevancy_sd"].iloc[0] == pytest.approx(0.3)


def test_a_question_with_no_usable_draw_has_no_spread():
    _record("faithfulness", "q1", "a1", [])
    spread = spread_columns(_results(("q1", "a1", "OK")))
    assert np.isnan(spread["faithfulness_sd"].iloc[0])
    assert spread["faithfulness_draws"].iloc[0] == 0


def test_a_single_usable_draw_has_no_spread():
    """Observed: two of three draws failed on the embedding server. The one
    left still gives the score, but says nothing about how much the judge moves."""
    _record("answer_relevancy", "q1", "a1", [0.845])
    spread = spread_columns(_results(("q1", "a1", "OK")))
    assert np.isnan(spread["answer_relevancy_sd"].iloc[0])
    assert spread["answer_relevancy_draws"].iloc[0] == 1


def test_a_failed_pipeline_row_carries_no_spread():
    _record("faithfulness", "q1", "ERROR", [0.0, 0.0])
    spread = spread_columns(_results(("q1", "ERROR", "ERROR")))
    assert spread.isna().all(axis=None)


def test_write_spread_appends_to_the_written_metrics(tmp_path):
    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame({"user_input": ["q1"], "faithfulness": [0.75]}).to_csv(metrics_csv, index=False)
    _record("faithfulness", "q1", "a1", [1.0, 0.5])
    frame = _results(("q1", "a1", "OK"))

    returned = write_spread(frame, metrics_csv)

    written = pd.read_csv(metrics_csv)
    assert written["faithfulness"].tolist() == [0.75]
    assert written["faithfulness_sd"].tolist() == pytest.approx([0.25])
    assert "faithfulness_sd" in returned.columns
    assert judge_repeats.DRAWS == {}, "draws of one profile must not leak into the next"


def test_the_metrics_keep_their_ragas_names():
    """Downstream code finds the scores by these column names."""
    assert judge_repeats.RepeatedFaithfulness(repeats=3).name == "faithfulness"
    relevancy = judge_repeats.RepeatedResponseRelevancy(repeats=3)
    assert relevancy.name == "answer_relevancy"
    assert relevancy.strictness == 3, "each draw is the metric as RAGAS defines it"


def test_draws_are_recorded_under_the_metric_question_and_answer():
    report = judge_repeats._recorder("faithfulness", SimpleNamespace(user_input="q", response="a"))
    report([0.5, 1.0], 0, None)
    assert judge_repeats.DRAWS[("faithfulness", "q", "a")] == [0.5, 1.0]
