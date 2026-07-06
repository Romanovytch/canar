"""Task 4 — YAML loader: maps core fields, preserves the rest as metadata, and
loads the real mini_client datasets when present."""

from pathlib import Path

import pytest
from dataset_loaders import load_dataset
from dataset_loaders.yaml_loader import load_yaml

BENCH_DIR = Path(__file__).resolve().parents[2]            # benchmark/
MINI_CLIENT = BENCH_DIR.parent / "benchmark_mini_client"   # sibling dir (gitignored)

SAMPLE = """
dataset_id: benchmark_demo_v0
document_group: demo_corpus
questions:
  - id: q1
    question: "What is X?"
    question_type: exact_lookup
    expected_answer: "X is Y."
    expected_source_files:
      - "corpus/a.qmd"
      - "corpus/b.qmd"
    expected_sections:
      - "Intro"
    expected_terms:
      - "X"
      - "Y"
    answer_present: true
    difficulty: easy
  - id: q2
    question: "Unanswerable?"
    question_type: no_answer
    expected_answer: "Not in the corpus."
    expected_source_files: []
    answer_present: false
    difficulty: medium
"""


def _write(tmp_path, text, name="ds.yaml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_maps_core_fields(tmp_path):
    items = load_yaml(_write(tmp_path, SAMPLE))
    assert len(items) == 2
    first = items[0]
    assert first.query == "What is X?"
    assert first.reference == "X is Y."
    # list of expected sources -> ';'-joined string for the retrieval metrics
    assert first.source_fiche == "corpus/a.qmd;corpus/b.qmd"


def test_preserves_metadata(tmp_path):
    first = load_yaml(_write(tmp_path, SAMPLE))[0]
    md = first.metadata
    assert md["id"] == "q1"
    assert md["question_type"] == "exact_lookup"
    assert md["expected_terms"] == ["X", "Y"]
    assert md["expected_sections"] == ["Intro"]
    assert md["answer_present"] is True
    assert md["difficulty"] == "easy"
    # dataset-level fields are attached to every item
    assert md["dataset_id"] == "benchmark_demo_v0"
    assert md["document_group"] == "demo_corpus"


def test_no_answer_question_has_empty_source(tmp_path):
    second = load_yaml(_write(tmp_path, SAMPLE))[1]
    assert second.source_fiche == ""
    assert second.metadata["answer_present"] is False


def test_dispatched_through_load_dataset(tmp_path):
    items = load_dataset(_write(tmp_path, SAMPLE))  # .yaml resolves to YAML loader
    assert items[0].query == "What is X?"


def test_yml_extension_also_works(tmp_path):
    items = load_dataset(_write(tmp_path, SAMPLE, name="ds.yml"))
    assert len(items) == 2


def test_missing_required_key_errors(tmp_path):
    bad = """
questions:
  - id: q1
    question: "no answer key here"
"""
    with pytest.raises(ValueError) as exc:
        load_yaml(_write(tmp_path, bad))
    assert "expected_answer" in str(exc.value)


def test_empty_file_errors(tmp_path):
    with pytest.raises(ValueError) as exc:
        load_yaml(_write(tmp_path, ""))
    assert "questions" in str(exc.value)


@pytest.mark.parametrize(
    "name", ["benchmark_utilitr_v0.yaml", "benchmark_dicovarg21_v0.yaml"]
)
def test_real_mini_client_datasets_load(name):
    path = MINI_CLIENT / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    items = load_yaml(path)
    assert len(items) > 0
    assert all(it.query and it.reference for it in items)
    # metadata is genuinely populated, not empty
    assert all(it.metadata for it in items)
