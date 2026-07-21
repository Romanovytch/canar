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
    - `DatasetSpec`                            (which dataset file; CSV or YAML)
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
from dataset_loaders import load_dataset
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.run_config import RunConfig


@dataclass
class PipelineOutput:
    """What one question yields when run through the pipeline under test."""
    answer: str
    contexts: list[str]                 # retrieved chunk texts (RAGAS context)
    paths: list[str] = field(default_factory=list)  # source paths, for retrieval metrics
    error: str | None = None                     # pipeline failure, if this question crashed
    retrieval_latency_s: float | None = None     # time spent retrieving
    generation_latency_s: float | None = None    # time spent generating the answer
    # Optional resource cost (only set when the resource flag is on; see
    # harness/resource_probe.py). Per phase for CPU/memory; GPU is device-level.
    # Per-strategy resource metrics only (#53): the signals that actually differ
    # between retrieval strategies. GPU is not here — it is run-level context
    # (same LLM for every strategy); see resource_probe.gpu_context().
    retrieval_cpu_s: float | None = None   # real in-process CPU of the method
    peak_rss_mb: float | None = None       # peak benchmark-process memory (turn)
    # Token usage (#64): prompt vs answer size — differs per strategy because the
    # retrieved context size differs.
    input_tokens: int | None = None        # tokens in the prompt sent to the LLM
    output_tokens: int | None = None       # tokens in the generated answer
    total_tokens: int | None = None        # input + output


# Per-question performance measurements carried on PipelineOutput. Each becomes a
# column only when at least one question reports a value, so a run without the
# resource flag adds no columns (latencies are always present, so they always do).
PERF_FIELDS = (
    "retrieval_latency_s",
    "generation_latency_s",
    "retrieval_cpu_s",
    "peak_rss_mb",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


def _pipeline_error_message(exc: Exception) -> str:
    """Make runtime failures obvious in terminal output and metrics files."""
    raw = f"{type(exc).__name__}: {exc}"
    lowered = raw.lower()
    if "outofmemoryerror" in lowered or "out of memory" in lowered:
        return f"GPU_OUT_OF_MEMORY: {raw}"
    return raw


ERROR_METRIC = "ERROR"
PIPELINE_ERROR_ANSWER = "[PIPELINE_ERROR]"


@dataclass
class DatasetSpec:
    """Where the dataset lives, plus a couple of run-level knobs.

    The file format (CSV / YAML) is detected from the extension, or forced with
    `fmt`. Field mapping lives in the format loaders (harness/dataset_loaders/),
    so this no longer carries column names.
    """
    path: Path
    fmt: str | None = None              # override format detection, e.g. "yaml"
    # None disables the retrieval metrics; also the label of the source column
    # written to the output.
    source_col: str | None = "source_fiche"
    limit: int | None = None            # None = all questions; int = quick subset


def _write_answers_md(path, df, title, tag, score_cols, source_col) -> None:
    """Write the text parts of a run as readable markdown, one section per question."""
    lines = [f"# {title}", ""]
    if tag:
        lines += ["> " + " · ".join(f"{k}: {v}" for k, v in tag.items()), ""]

    for i, row in df.iterrows():
        lines += [f"## Q{i + 1}. {row['user_input']}", ""]

        score_parts = []
        for c in score_cols:
            if c not in df.columns or pd.isna(row[c]):
                continue
            value = row[c]
            if isinstance(value, (int, float)):
                score_parts.append(f"{c}={value:.2f}")
            else:
                score_parts.append(f"{c}={value}")
        scores = " · ".join(score_parts)
        if scores:
            lines += [f"`{scores}`", ""]

        if row.get("pipeline_status") == "ERROR":
            lines += ["**Pipeline error**", "", str(row.get("pipeline_error", "")), ""]

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
    group_dir: Path | None = None,
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
    items = load_dataset(dataset.path, fmt=dataset.fmt)
    if dataset.limit:
        items = items[:dataset.limit]
    track_hits = dataset.source_col is not None
    print(f"{name}: {len(items)} questions\n")

    samples, retr_rows, all_paths = [], [], []
    pipeline_statuses, pipeline_errors = [], []
    metadatas = []                       # per-question extras (rich YAML datasets)
    perf = {perf_field: [] for perf_field in PERF_FIELDS}  # per-question latency/resource values
    metric_names = tuple(retrieval_metrics.compute([], "").keys())
    for item in items:
        question = item.query
        print(f"Q: {question}")
        try:
            out = pipeline(question)
        except Exception as e:  # keep the run going, but make the failure impossible to miss
            error = _pipeline_error_message(e)
            print(f"   [PIPELINE_ERROR: {error}]")
            out = PipelineOutput(PIPELINE_ERROR_ANSWER, [], [], error=error)

        failed = out.error is not None
        pipeline_statuses.append("ERROR" if failed else "OK")
        pipeline_errors.append(out.error or "")

        metadatas.append(item.metadata)
        for perf_field in PERF_FIELDS:
            perf[perf_field].append(getattr(out, perf_field))
        if track_hits:
            # deterministic retrieval metrics (Hit Rate@k, MRR, Recall@k, ...)
            if failed:
                m = {metric_name: ERROR_METRIC for metric_name in metric_names}
                retr_rows.append(m)
                all_paths.append("PIPELINE_ERROR")
                print(f"   retrieval=ERROR | A: {out.answer}\n")
            else:
                m = retrieval_metrics.compute(out.paths, item.source_fiche, k=retrieval_k)
                retr_rows.append(m)
                all_paths.append("; ".join(out.paths))
                print(f"   hit={m['hit_rate']:.0f} mrr={m['mrr']:.2f} "
                      f"recall={m['recall']:.2f} | A: {out.answer[:60]}...\n")
        else:
            status = "ERROR" if failed else "OK"
            print(f"   status={status} | A: {out.answer[:80]}...\n")

        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=out.contexts,
            response=out.answer,
            reference=item.reference,
        ))

    if track_hits and retr_rows:
        ok_rows = [r for r, status in zip(retr_rows, pipeline_statuses) if status == "OK"]
        if ok_rows:
            hit_rate = sum(r["hit_rate"] for r in ok_rows) / len(ok_rows)
            mrr = sum(r["mrr"] for r in ok_rows) / len(ok_rows)
            print(f"Retrieval — Hit Rate@k: {hit_rate:.0%} | MRR: {mrr:.3f}\n")
        failed = len(retr_rows) - len(ok_rows)
        if failed:
            print(f"Retrieval — {failed} question(s) failed with PIPELINE_ERROR\n")

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
    out_df["pipeline_status"] = pipeline_statuses
    out_df["pipeline_error"] = pipeline_errors
    meta_cols |= {"pipeline_status", "pipeline_error"}
    if tag:  # constant columns (e.g. profile name) — kept out of the score means
        for key, value in tag.items():
            out_df[key] = value
        meta_cols |= set(tag)
    # Latency/resource columns: emit one per field that has any value. Latencies
    # are always present; resource fields only when the resource flag was on.
    for perf_field in PERF_FIELDS:
        values = perf[perf_field]
        if any(v is not None for v in values):
            out_df[perf_field] = values
    if track_hits:
        # one column per retrieval metric (hit_rate, mrr, recall, precision, ndcg)
        for metric_name in retr_rows[0]:
            out_df[metric_name] = [r[metric_name] for r in retr_rows]
        out_df["retrieved_paths"] = all_paths
        out_df[dataset.source_col] = [item.source_fiche for item in items]
        meta_cols |= {"retrieved_paths", dataset.source_col}

    # Surface per-question metadata from richer (YAML) datasets as columns, so the
    # extra signal — question_type, difficulty, answer_present, id, ... — reaches
    # the reporting layer and runs can be sliced by it. Only scalar values become
    # columns (lists like expected_terms don't fit a CSV cell); they stay on
    # DatasetItem.metadata for deeper use. CSV datasets carry no metadata, so this
    # is a no-op for them and metrics.csv is unchanged.
    scalar = (str, int, float, bool, type(None))
    meta_keys: list[str] = []
    for md in metadatas:
        for key, value in md.items():
            if key not in meta_keys and isinstance(value, scalar):
                meta_keys.append(key)
    for key in meta_keys:
        out_df[key] = [md.get(key) for md in metadatas]
    meta_cols |= set(meta_keys)

    score_cols = [c for c in out_df.columns if c not in meta_cols]
    error_mask = out_df["pipeline_status"] == "ERROR"
    if error_mask.any():
        out_df[score_cols] = out_df[score_cols].astype(object)
        out_df.loc[error_mask, score_cols] = ERROR_METRIC
    display_cols = ["user_input", "pipeline_status", "pipeline_error"] + score_cols
    print(out_df[display_cols].to_string(index=False))
    numeric_scores = out_df[score_cols].apply(pd.to_numeric, errors="coerce")
    print("\nMean scores:")
    print(numeric_scores.mean(numeric_only=True).to_string())

    # Output folder, holding the numbers and the text split apart:
    #   metrics.csv  — only the scores (plus a row id and provenance), easy to read/plot
    #   answers.md   — the text parts (question, answer, reference, retrieved context)
    # With `group_dir`, all strategies of one benchmark run share it and each
    # gets a subfolder (group_dir/<strategy>/); otherwise a timestamped folder.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if group_dir is not None:
        out_dir = group_dir / (file_label or "run")
    else:
        results_dir.mkdir(exist_ok=True)
        label = f"{file_label}_" if file_label else ""
        out_dir = results_dir / f"{label}{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag_cols = list(tag) if tag else []
    extra_cols = [k for k in meta_keys if k not in tag_cols]  # dataset metadata
    metrics_path = out_dir / "metrics.csv"
    status_cols = ["pipeline_status", "pipeline_error"]
    out_df[["user_input"] + tag_cols + extra_cols + status_cols + score_cols].to_csv(
        metrics_path,
        index=False,
    )

    answers_path = out_dir / "answers.md"
    _write_answers_md(answers_path, out_df, name, tag, score_cols, dataset.source_col)

    print(f"\nMetrics saved to {metrics_path}")
    print(f"Answers saved to {answers_path}")
    return out_df
