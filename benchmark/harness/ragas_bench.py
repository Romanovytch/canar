"""
Reusable RAGAS benchmark harness.

A benchmark = a dataset of questions + a `pipeline` callable that answers one
question the way the product (or a replica) does. Everything that repeats
across benchmarks lives here:

    - the per-question loop (retrieve -> generate -> deterministic retrieval metrics)
    - the RAGAS evaluation call
    - assembling, printing and saving the timestamped results CSV

What changes between benchmarks stays in the thin caller script:

    - `pipeline(question) -> PipelineOutput`   (how an answer is produced)
    - `DatasetSpec`                            (which CSV, which columns)
    - the metrics list and the judge LLM/embeddings

To build a NEW benchmark, write a short script that defines a `pipeline`
function and calls `run_benchmark(...)`. See e2e/eval_e2e.py for a full,
product-grade example (CanaR + AgoRa).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import retrieval_metrics
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.run_config import RunConfig


@dataclass
class PipelineOutput:
    """What one question yields when run through the pipeline under test."""
    answer: str
    contexts: list[str]                 # retrieved chunk texts (RAGAS context)
    paths: list[str] = field(default_factory=list)  # source paths, for retrieval metrics
    retrieval_latency_s: float | None = None         # time spent retrieving (Latency metric)


@dataclass
class DatasetSpec:
    """Where the questions live and which columns to read."""
    path: Path
    question_col: str = "query"
    reference_col: str = "grading_notes"
    # column holding the expected source file; None disables the retrieval metrics
    source_col: str | None = "source_fiche"
    limit: int | None = None            # None = all rows; int = quick subset


def retrieval_hit(expected: str, paths: list[str]) -> bool:
    """Deterministic: did the expected source file appear among retrieved paths?"""
    return any(expected in p or p.endswith(Path(expected).name) for p in paths)


def _write_answers_md(path, df, title, tag, score_cols, source_col) -> None:
    """Write the text parts of a run as readable markdown, one section per question."""
    lines = [f"# {title}", ""]
    if tag:
        lines += ["> " + " · ".join(f"{k}: {v}" for k, v in tag.items()), ""]

    for i, row in df.iterrows():
        lines += [f"## Q{i + 1}. {row['user_input']}", ""]

        scores = " · ".join(
            f"{c}={row[c]:.2f}"
            for c in score_cols
            if c in df.columns and pd.notna(row[c])
        )
        if scores:
            lines += [f"`{scores}`", ""]

        answer = str(row.get("response", "")).strip()
        lines += ["**Answer**", "", answer or "_(empty)_", ""]

        if "reference" in df.columns:
            lines += ["**Reference**", "", str(row["reference"]).strip(), ""]

        if source_col and source_col in df.columns:
            lines += [f"**Expected source:** `{row[source_col]}`", ""]
        if "retrieved_paths" in df.columns:
            lines += [f"**Retrieved:** `{row['retrieved_paths']}`", ""]

        contexts = row.get("retrieved_contexts")
        if contexts is not None and len(contexts) > 0:
            lines += ["**Retrieved context**", ""]
            for j, ctx in enumerate(contexts, 1):
                lines += [
                    f"<details><summary>Source {j}</summary>",
                    "",
                    str(ctx),
                    "",
                    "</details>",
                    "",
                ]

        lines += ["---", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    *,
    name: str,
    dataset: DatasetSpec,
    pipeline: Callable[[str], PipelineOutput],
    metrics: list,
    judge_llm,
    judge_embeddings,
    results_dir: Path,
    run_config: RunConfig | None = None,
    retrieval_k: int | None = None,
    tag: dict | None = None,
    file_label: str | None = None,
) -> pd.DataFrame:
    """
    Run `pipeline` over every question in `dataset`, score with RAGAS *and* the
    deterministic retrieval metrics, and save a timestamped CSV in
    `results_dir`. Returns the results DataFrame.

    retrieval_k : cutoff for the @k retrieval metrics; None = whatever the
                  pipeline retrieved (CanaR's configured top_k).
    tag         : constant columns added to every row (e.g. {"profile": "..."}),
                  so several runs can be concatenated and compared.
    file_label  : prefix for the output CSV name (e.g. the profile name).
    """
    df = pd.read_csv(dataset.path)
    if dataset.limit:
        df = df.head(dataset.limit)
    track_hits = dataset.source_col is not None
    print(f"{name}: {len(df)} questions\n")

    samples, retr_rows, all_paths, latencies = [], [], [], []
    for _, row in df.iterrows():
        question = row[dataset.question_col]
        print(f"Q: {question}")
        try:
            out = pipeline(question)
        except Exception as e:  # one bad question shouldn't kill the whole run
            print(f"   [pipeline failed: {e}]")
            out = PipelineOutput("", [], [])

        latencies.append(out.retrieval_latency_s)
        if track_hits:
            # deterministic retrieval metrics (Hit Rate@k, MRR, Recall@k, ...)
            m = retrieval_metrics.compute(out.paths, row[dataset.source_col], k=retrieval_k)
            retr_rows.append(m)
            all_paths.append("; ".join(out.paths))
            print(f"   hit={m['hit_rate']:.0f} mrr={m['mrr']:.2f} "
                  f"recall={m['recall']:.2f} | A: {out.answer[:60]}...\n")
        else:
            print(f"   A: {out.answer[:80]}...\n")

        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=out.contexts,
            response=out.answer,
            reference=row[dataset.reference_col],
        ))

    if track_hits and retr_rows:
        hit_rate = sum(r["hit_rate"] for r in retr_rows) / len(retr_rows)
        mrr = sum(r["mrr"] for r in retr_rows) / len(retr_rows)
        print(f"Retrieval — Hit Rate@k: {hit_rate:.0%} | MRR: {mrr:.3f}\n")

    print("Running RAGAS evaluation (slow on a local judge) ...\n")
    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config or RunConfig(timeout=600, max_workers=1),
    )

    out_df = results.to_pandas()
    meta_cols = {"user_input", "retrieved_contexts", "response", "reference"}
    if tag:  # constant columns (e.g. profile name) — kept out of the score means
        for key, value in tag.items():
            out_df[key] = value
        meta_cols |= set(tag)
    # latency is always available (deterministic, no source column needed)
    if any(latency is not None for latency in latencies):
        out_df["retrieval_latency_s"] = latencies
    if track_hits:
        # one column per retrieval metric (hit_rate, mrr, recall, precision, ndcg)
        for metric_name in retr_rows[0]:
            out_df[metric_name] = [r[metric_name] for r in retr_rows]
        out_df["retrieved_paths"] = all_paths
        out_df[dataset.source_col] = df[dataset.source_col].values
        meta_cols |= {"retrieved_paths", dataset.source_col}

    score_cols = [c for c in out_df.columns if c not in meta_cols]
    print(out_df[["user_input"] + score_cols].to_string(index=False))
    print(f"\nMean scores:\n{out_df[score_cols].mean(numeric_only=True).to_string()}")

    # Each run gets its own folder with the numbers and the text split apart:
    #   metrics.csv  — only the scores (plus a row id and provenance), easy to read/plot
    #   answers.md   — the text parts (question, answer, reference, retrieved context)
    results_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{file_label}_" if file_label else ""
    run_dir = results_dir / f"{label}{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tag_cols = list(tag) if tag else []
    metrics_path = run_dir / "metrics.csv"
    out_df[["user_input"] + tag_cols + score_cols].to_csv(metrics_path, index=False)

    answers_path = run_dir / "answers.md"
    _write_answers_md(answers_path, out_df, name, tag, score_cols, dataset.source_col)

    print(f"\nMetrics saved to {metrics_path}")
    print(f"Answers saved to {answers_path}")
    return out_df
