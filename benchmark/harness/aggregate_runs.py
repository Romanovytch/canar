"""Combine several executions of the same configuration into one table.

Each execution writes its own directory, so three runs of a configuration leave
three independent `comparison.csv` files and no combined view. What the answer
quality scores need is the opposite: a central value per profile and the spread
around it, since a difference between profiles is only meaningful when it is
larger than the judge's own variation between runs.

Two properties of the data shape the aggregation.

A question the judge failed to score is absent rather than zero, and it is not
the same question every time — the embedding server drops some, the judge
returns unparseable output for others. Averaging whatever happens to be present
compares means taken over different question sets, which moved a profile's
answer relevancy by 0.1 between runs that were otherwise identical. Only
questions scored in *every* run are used, per metric, since the two metrics fail
independently.

A question the dataset declares no source for is left unscored by the retrieval
metrics on purpose. Those arrive as blanks and are skipped the same way.

    python -m aggregate_runs e2e/results/run_A e2e/results/run_B ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Scores worth combining. Retrieval metrics and submitted tokens are
# deterministic and will show a spread of zero, which is itself worth seeing.
METRICS = (
    "faithfulness",
    "answer_relevancy",
    "hit_rate",
    "mrr",
    "recall",
    "precision",
    "ndcg",
    "input_tokens",
    "output_tokens",
    "retrieval_latency_s",
    "generation_latency_s",
)
QUESTION_KEY = "id"


def load_runs(run_dirs: list[Path]) -> pd.DataFrame:
    """Every per-question row from every profile of every run, tagged with `run`."""
    frames = []
    for run_dir in run_dirs:
        for metrics_csv in sorted(run_dir.glob("*/metrics.csv")):
            frame = pd.read_csv(metrics_csv)
            frame["run"] = run_dir.name
            if "profile" not in frame.columns:
                frame["profile"] = metrics_csv.parent.name
            frames.append(frame)
    if not frames:
        raise SystemExit(f"No */metrics.csv found under {[str(d) for d in run_dirs]}")
    combined = pd.concat(frames, ignore_index=True)
    if "pipeline_status" in combined.columns:
        combined = combined[combined["pipeline_status"] != "ERROR"]
    if QUESTION_KEY not in combined.columns:
        raise SystemExit(
            f"Rows carry no {QUESTION_KEY!r} column, so questions cannot be matched "
            "across runs. The YAML datasets provide it; CSV datasets do not."
        )
    return combined


def questions_scored_in_every_run(frame: pd.DataFrame, metric: str) -> set:
    """Question ids this metric has a value for in each of the runs present."""
    scored = frame[pd.to_numeric(frame[metric], errors="coerce").notna()]
    per_run = scored.groupby("run")[QUESTION_KEY].apply(set)
    if per_run.empty:
        return set()
    return set.intersection(*per_run)


def aggregate(frame: pd.DataFrame, metrics=METRICS) -> pd.DataFrame:
    """One row per profile: mean over runs, and the spread across them."""
    rows = []
    for profile, profile_rows in frame.groupby("profile"):
        row = {"profile": profile, "runs": profile_rows["run"].nunique()}
        for metric in metrics:
            if metric not in profile_rows.columns:
                continue
            common = questions_scored_in_every_run(profile_rows, metric)
            usable = profile_rows[profile_rows[QUESTION_KEY].isin(common)]
            per_run = (
                usable.assign(**{metric: pd.to_numeric(usable[metric], errors="coerce")})
                .groupby("run")[metric]
                .mean()
            )
            row[f"{metric}_n"] = len(common)
            row[f"{metric}_mean"] = per_run.mean() if len(per_run) else float("nan")
            # ddof=0: these runs are the population being described, not a sample
            # drawn from a larger one.
            row[f"{metric}_sd"] = per_run.std(ddof=0) if len(per_run) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("profile").reset_index(drop=True)


def paired_table(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Question by profile, averaged over runs — the shape a paired test needs."""
    common = questions_scored_in_every_run(frame, metric)
    usable = frame[frame[QUESTION_KEY].isin(common)].copy()
    usable[metric] = pd.to_numeric(usable[metric], errors="coerce")
    return usable.pivot_table(
        index=QUESTION_KEY, columns="profile", values=metric, aggfunc="mean"
    )


def _format(table: pd.DataFrame, metrics=METRICS) -> str:
    lines = []
    present = [m for m in metrics if f"{m}_mean" in table.columns]
    header = f"{'profile':<38} {'runs':>4}  " + "  ".join(f"{m:>22}" for m in present)
    lines += [header, "-" * len(header)]
    for _, row in table.iterrows():
        cells = []
        for metric in present:
            mean, sd, n = row[f"{metric}_mean"], row[f"{metric}_sd"], row[f"{metric}_n"]
            cells.append(f"{mean:8.3f} ± {sd:<6.3f} n={n:<3.0f}"[:22].rjust(22))
        lines.append(f"{row['profile']:<38} {row['runs']:>4}  " + "  ".join(cells))
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    frame = load_runs([Path(a) for a in argv])
    table = aggregate(frame)
    print(_format(table))
    print()
    print(f"{table['runs'].max()} run(s) combined. "
          "n is the questions scored in every run, per metric; a metric whose n is "
          "below the dataset size lost questions the judge could not score.")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
