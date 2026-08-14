"""Task 6 — run_benchmark consumes the loader output (not a raw CSV): it runs
for both CSV and YAML, and YAML metadata reaches the saved metrics.csv. RAGAS is
stubbed so the test needs no judge LLM."""

import pandas as pd
import pytest
import ragas_bench
from ragas_bench import DatasetSpec, PipelineOutput, run_benchmark


class _FakeResult:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


@pytest.fixture
def stub_ragas(monkeypatch):
    """Replace ragas' EvaluationDataset + evaluate so no judge LLM is needed.

    EvaluationDataset becomes identity (returns the sample list); evaluate
    returns one row per sample with a single dummy score column.
    """
    monkeypatch.setattr(ragas_bench, "EvaluationDataset", lambda *, samples: samples)

    def fake_evaluate(*, dataset, metrics, llm, embeddings, run_config):
        samples = dataset  # the identity EvaluationDataset above
        df = pd.DataFrame(
            {
                "user_input": [s.user_input for s in samples],
                "retrieved_contexts": [s.retrieved_contexts for s in samples],
                "response": [s.response for s in samples],
                "reference": [s.reference for s in samples],
                "faithfulness": [1.0] * len(samples),
            }
        )
        return _FakeResult(df)

    monkeypatch.setattr(ragas_bench, "evaluate", fake_evaluate)


def _pipeline(question):
    return PipelineOutput(answer="ans", contexts=["ctx"], paths=["some/path.qmd"])


def _run(dataset_path, tmp_path, label):
    return run_benchmark(
        name="t",
        dataset=DatasetSpec(path=dataset_path),
        pipeline=_pipeline,
        metrics=[],
        judge_llm=None,
        judge_embeddings=None,
        results_dir=tmp_path,
        file_label=label,
        group_dir=tmp_path,  # deterministic out dir: tmp_path/<label>/
    )


def test_csv_dataset_runs_with_no_metadata_columns(stub_ragas, tmp_path):
    csv = tmp_path / "ds.csv"
    csv.write_text(
        "query,grading_notes,source_fiche\nq1,r1,some/path.qmd\n", encoding="utf-8"
    )
    out_df = _run(csv, tmp_path, "csv")

    assert list(out_df["user_input"]) == ["q1"]
    assert "hit_rate" in out_df.columns          # retrieval metrics computed
    assert out_df["hit_rate"].iloc[0] == 1.0     # path matched the expected source

    # No regression: a CSV carries no metadata, so metrics.csv gains no extra cols.
    metrics = pd.read_csv(tmp_path / "csv" / "metrics.csv")
    assert "question_type" not in metrics.columns


def test_pipeline_failure_is_saved_as_error_not_zero(stub_ragas, tmp_path):
    csv = tmp_path / "ds.csv"
    csv.write_text(
        "query,grading_notes,source_fiche\nq1,r1,some/path.qmd\n", encoding="utf-8"
    )

    def failing_pipeline(question):
        raise RuntimeError("boom")

    out_df = run_benchmark(
        name="t",
        dataset=DatasetSpec(path=csv),
        pipeline=failing_pipeline,
        metrics=[],
        judge_llm=None,
        judge_embeddings=None,
        results_dir=tmp_path,
        file_label="err",
        group_dir=tmp_path,
    )

    assert out_df["pipeline_status"].iloc[0] == "ERROR"
    assert "RuntimeError: boom" in out_df["pipeline_error"].iloc[0]
    assert out_df["response"].iloc[0] == "[PIPELINE_ERROR]"
    assert out_df["hit_rate"].iloc[0] == "ERROR"

    metrics = pd.read_csv(tmp_path / "err" / "metrics.csv")
    assert metrics["pipeline_status"].iloc[0] == "ERROR"
    assert metrics["hit_rate"].iloc[0] == "ERROR"


def test_yaml_metadata_reaches_metrics_csv(stub_ragas, tmp_path):
    yaml_text = """
dataset_id: d
document_group: g
questions:
  - id: q1
    question: "What is X?"
    question_type: exact_lookup
    expected_answer: "X is Y."
    expected_source_files: ["some/path.qmd"]
    difficulty: easy
"""
    ds = tmp_path / "ds.yaml"
    ds.write_text(yaml_text, encoding="utf-8")
    out_df = _run(ds, tmp_path, "yaml")

    # scalar metadata surfaced as columns on the result frame
    for col in ("id", "question_type", "difficulty", "dataset_id", "document_group"):
        assert col in out_df.columns
    assert out_df["question_type"].iloc[0] == "exact_lookup"

    # and persisted to metrics.csv (the reporting layer)
    metrics = pd.read_csv(tmp_path / "yaml" / "metrics.csv")
    assert metrics["question_type"].iloc[0] == "exact_lookup"
    assert metrics["difficulty"].iloc[0] == "easy"


def test_empty_expected_sources_are_nan_and_excluded_from_summary(
    stub_ragas, tmp_path, capsys
):
    yaml_text = """
dataset_id: d
questions:
  - id: scored
    question: "Scored question?"
    expected_answer: "Answer."
    expected_source_files: ["some/path.qmd"]
  - id: unscored
    question: "Question with no relevant source?"
    expected_answer: "No answer in the corpus."
    expected_source_files: []
"""
    ds = tmp_path / "ds.yaml"
    ds.write_text(yaml_text, encoding="utf-8")

    out_df = _run(ds, tmp_path, "empty-sources")
    retrieval_cols = ["hit_rate", "mrr", "recall", "precision", "ndcg"]
    unscored = out_df.loc[out_df["id"] == "unscored"].iloc[0]

    assert unscored[retrieval_cols].isna().all()
    output = capsys.readouterr().out
    assert "Retrieval — Hit Rate@k: 100% | MRR: 1.000" in output
    assert "Retrieval — 1 question(s) excluded (no expected source)" in output

    metrics = pd.read_csv(tmp_path / "empty-sources" / "metrics.csv")
    saved_unscored = metrics.loc[metrics["id"] == "unscored"].iloc[0]
    assert saved_unscored[retrieval_cols].isna().all()


RESOURCE_COLS = (
    "retrieval_cpu_s",
    "peak_rss_mb",
)


def test_resource_fields_become_columns(stub_ragas, tmp_path):
    csv = tmp_path / "ds.csv"
    csv.write_text(
        "query,grading_notes,source_fiche\nq1,r1,some/path.qmd\n", encoding="utf-8"
    )

    def pipeline_with_resources(question):
        return PipelineOutput(
            answer="ans",
            contexts=["ctx"],
            paths=["some/path.qmd"],
            retrieval_latency_s=0.1,
            generation_latency_s=0.2,
            retrieval_cpu_s=0.05,
            peak_rss_mb=120.0,
            input_tokens=300,
            output_tokens=40,
            total_tokens=340,
        )

    out_df = run_benchmark(
        name="t",
        dataset=DatasetSpec(path=csv),
        pipeline=pipeline_with_resources,
        metrics=[],
        judge_llm=None,
        judge_embeddings=None,
        results_dir=tmp_path,
        file_label="res",
        group_dir=tmp_path,
    )
    for col in RESOURCE_COLS + ("input_tokens", "output_tokens", "total_tokens"):
        assert col in out_df.columns
    metrics = pd.read_csv(tmp_path / "res" / "metrics.csv")
    assert metrics["retrieval_cpu_s"].iloc[0] == 0.05
    assert metrics["peak_rss_mb"].iloc[0] == 120.0
    assert metrics["total_tokens"].iloc[0] == 340


def test_no_resource_fields_no_columns(stub_ragas, tmp_path):
    # The default _pipeline reports no latency/resource values (flag off).
    csv = tmp_path / "ds.csv"
    csv.write_text(
        "query,grading_notes,source_fiche\nq1,r1,some/path.qmd\n", encoding="utf-8"
    )
    out_df = _run(csv, tmp_path, "nores")
    for col in RESOURCE_COLS:
        assert col not in out_df.columns
