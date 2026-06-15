"""
Reusable RAGAS benchmark harness.

A benchmark = a dataset of questions + a `pipeline` callable that answers one
question the way the product (or a replica) does. Everything that repeats
across benchmarks lives here:

    - the per-question loop (retrieve -> generate -> deterministic retrieval_hit)
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
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.run_config import RunConfig


@dataclass
class PipelineOutput:
    """What one question yields when run through the pipeline under test."""
    answer: str
    contexts: list[str]                 # retrieved chunk texts (RAGAS context)
    paths: list[str] = field(default_factory=list)  # source paths, for retrieval_hit


@dataclass
class DatasetSpec:
    """Where the questions live and which columns to read."""
    path: Path
    question_col: str = "query"
    reference_col: str = "grading_notes"
    # column holding the expected source file; None disables the retrieval_hit metric
    source_col: str | None = "source_fiche"
    limit: int | None = None            # None = all rows; int = quick subset


def retrieval_hit(expected: str, paths: list[str]) -> bool:
    """Deterministic: did the expected source file appear among retrieved paths?"""
    return any(expected in p or p.endswith(Path(expected).name) for p in paths)


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
) -> pd.DataFrame:
    """
    Run `pipeline` over every question in `dataset`, score with RAGAS, and
    save a timestamped CSV in `results_dir`. Returns the results DataFrame.
    """
    df = pd.read_csv(dataset.path)
    if dataset.limit:
        df = df.head(dataset.limit)
    track_hits = dataset.source_col is not None
    print(f"{name}: {len(df)} questions\n")

    samples, hits, all_paths = [], [], []
    for _, row in df.iterrows():
        question = row[dataset.question_col]
        print(f"Q: {question}")
        try:
            out = pipeline(question)
        except Exception as e:  # one bad question shouldn't kill the whole run
            print(f"   [pipeline failed: {e}]")
            out = PipelineOutput("", [], [])

        if track_hits:
            expected = row[dataset.source_col]
            hit = retrieval_hit(expected, out.paths)
            hits.append(hit)
            all_paths.append("; ".join(out.paths))
            print(f"   retrieval_hit={'YES' if hit else 'NO'} | A: {out.answer[:80]}...\n")
        else:
            print(f"   A: {out.answer[:80]}...\n")

        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=out.contexts,
            response=out.answer,
            reference=row[dataset.reference_col],
        ))

    if track_hits and hits:
        print(f"Retrieval hit rate: {sum(hits)}/{len(hits)} = {sum(hits)/len(hits):.0%}\n")

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
    if track_hits:
        out_df["retrieval_hit"] = hits
        out_df["retrieved_paths"] = all_paths
        out_df[dataset.source_col] = df[dataset.source_col].values
        meta_cols |= {"retrieved_paths", dataset.source_col}

    score_cols = [c for c in out_df.columns if c not in meta_cols]
    print(out_df[["user_input"] + score_cols].to_string(index=False))
    print(f"\nMean scores:\n{out_df[score_cols].mean(numeric_only=True).to_string()}")

    results_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = results_dir / f"results_{stamp}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")
    return out_df
