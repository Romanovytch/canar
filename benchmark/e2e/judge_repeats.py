"""Judging each question several times, to see how much the RAGAS score moves.

The judge is an LLM, so faithfulness and answer relevancy change between runs on
byte-identical answers (see REPRODUCIBILITY.md). With `judge_repeats: N`, each
metric scores every question N times. The recorded score is the mean of the
draws, and the spread of the draws is written next to it, per question, so the
noise of the judge can be read off the results instead of guessed.

Retrieval and generation still run once: the answer being judged is fixed and
only the measurement repeats.

Kept out of harness/: the harness is product-agnostic and calls RAGAS once, so
the repetition belongs to the metric objects the e2e benchmark hands it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from ragas.metrics import Faithfulness, ResponseRelevancy

JUDGED_METRICS = ("faithfulness", "answer_relevancy")

# Draws of the profile being judged, keyed by (metric, question, answer). Filled
# while RAGAS scores, read back by `write_spread` once the profile is done.
DRAWS: dict[tuple[str, str, str], list[float]] = {}


async def mean_of_draws(score, sample, callbacks, repeats, report=None):
    """Average `repeats` judgements of one question, ignoring the ones that failed.

    A draw fails two ways: RAGAS *returns* NaN when it decides it cannot score,
    and *raises* when the judge's reply does not parse or a server call fails.
    Both cost one draw, not the question. When no draw produced a number the
    last exception is re-raised, so an unscorable question fails exactly as it
    does without repetition.

    `report` receives (valid_draws, failed_count, last_error).
    """
    valid = []
    failed = 0
    error = None
    for _ in range(repeats):
        try:
            draw = await score(sample, callbacks)
        except Exception as exc:  # noqa: BLE001 — any judge failure is one lost draw
            error, failed = exc, failed + 1
            continue
        if draw is None or np.isnan(draw):
            failed += 1
        else:
            valid.append(float(draw))
    if report is not None and repeats > 1:
        report(valid, failed, error)
    if not valid:
        if error is not None:
            raise error
        return float("nan")
    return float(np.mean(valid))


def _recorder(metric: str, sample):
    """Keep the draws for `write_spread` and show them as the run goes."""

    def report(valid, failed, error):
        DRAWS[(metric, sample.user_input, sample.response)] = valid
        shown = ", ".join(f"{d:.3f}" for d in valid) or "-"
        note = ""
        if failed:
            reason = f": {type(error).__name__}" if error is not None else ""
            note = f", {failed} failed{reason}"
        sd = f"{np.std(valid):.3f}" if len(valid) > 1 else "-"
        print(f"   {metric} draws [{shown}] sd={sd}{note}")

    return report


@dataclass
class RepeatedFaithfulness(Faithfulness):
    """Faithfulness scored `repeats` times per question; the mean is the score."""

    repeats: int = 1

    async def _single_turn_ascore(self, sample, callbacks):
        return await mean_of_draws(
            super()._single_turn_ascore,
            sample,
            callbacks,
            self.repeats,
            report=_recorder(self.name, sample),
        )


@dataclass
class RepeatedResponseRelevancy(ResponseRelevancy):
    """Answer relevancy scored `repeats` times per question; the mean is the score.

    Each draw is the metric exactly as RAGAS defines it — `strictness` questions
    generated from the answer, averaged — so repeating it does not change what
    it measures, only how precisely.
    """

    repeats: int = 1

    async def _single_turn_ascore(self, sample, callbacks):
        return await mean_of_draws(
            super()._single_turn_ascore,
            sample,
            callbacks,
            self.repeats,
            report=_recorder(self.name, sample),
        )


def spread_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Per question: standard deviation of the draws and how many were usable.

    The spread needs at least two usable draws: one draw has a spread of zero
    only arithmetically, and reading that as "the judge was stable" would be
    wrong, so it stays blank. Rows the pipeline failed on carry no judgement
    either.
    """
    columns = {}
    for metric in JUDGED_METRICS:
        draws = [
            DRAWS.get((metric, question, answer), [])
            for question, answer in zip(frame["user_input"], frame["response"], strict=True)
        ]
        # ddof=0: the draws taken are the population being described.
        columns[f"{metric}_sd"] = [float(np.std(d)) if len(d) > 1 else np.nan for d in draws]
        columns[f"{metric}_draws"] = [len(d) for d in draws]
    spread = pd.DataFrame(columns, index=frame.index)
    if "pipeline_status" in frame.columns:
        spread.loc[frame["pipeline_status"] == "ERROR"] = np.nan
    return spread


def write_spread(frame: pd.DataFrame, metrics_csv: Path) -> pd.DataFrame:
    """Add the spread to a profile's results, in memory and in its metrics.csv.

    The harness has already written metrics.csv from these same rows, in this
    same order, so the columns are appended rather than recomputed.
    """
    spread = spread_columns(frame)
    written = pd.read_csv(metrics_csv)
    written = pd.concat([written, spread.reset_index(drop=True)], axis=1)
    written.to_csv(metrics_csv, index=False)
    DRAWS.clear()
    return pd.concat([frame, spread], axis=1)
