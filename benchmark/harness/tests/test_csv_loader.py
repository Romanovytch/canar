"""Task 3 — CSV loader: no regression vs. the columns ragas_bench used, and
the real benchmark dataset still loads cleanly."""

from pathlib import Path

import pytest
from dataset_loaders import load_dataset
from dataset_loaders.csv_loader import load_csv

BENCH_DIR = Path(__file__).resolve().parents[2]  # benchmark/
REAL_DATASET = BENCH_DIR / "datasets" / "utilitr_questions.csv"


def _write_csv(tmp_path, text):
    p = tmp_path / "ds.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_maps_core_columns(tmp_path):
    csv = _write_csv(
        tmp_path,
        "query,grading_notes,source_fiche\n"
        '"How to import CSV?","use read_delim","fiche_a.qmd"\n',
    )
    items = load_csv(csv)
    assert len(items) == 1
    item = items[0]
    assert item.query == "How to import CSV?"
    assert item.reference == "use read_delim"
    assert item.source_fiche == "fiche_a.qmd"
    assert item.metadata == {}  # CSV carries no extra metadata


def test_dispatched_through_load_dataset(tmp_path):
    csv = _write_csv(
        tmp_path, "query,grading_notes,source_fiche\nq,r,s.qmd\n"
    )
    items = load_dataset(csv)  # .csv resolves to the CSV loader
    assert items[0].query == "q"


def test_missing_required_column_errors(tmp_path):
    csv = _write_csv(tmp_path, "query,source_fiche\nq,s.qmd\n")  # no grading_notes
    with pytest.raises(ValueError) as exc:
        load_csv(csv)
    assert "grading_notes" in str(exc.value)


def test_missing_source_column_is_tolerated(tmp_path):
    csv = _write_csv(tmp_path, "query,grading_notes\nq,r\n")
    items = load_csv(csv)
    assert items[0].source_fiche == ""


@pytest.mark.skipif(not REAL_DATASET.exists(), reason="real dataset not present")
def test_real_dataset_loads():
    items = load_csv(REAL_DATASET)
    assert len(items) > 0
    assert all(it.query and it.reference for it in items)
