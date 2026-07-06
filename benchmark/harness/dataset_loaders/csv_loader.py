"""
CSV dataset loader — preserves the benchmark's current behavior.

Reads the same columns ``ragas_bench`` read via ``DatasetSpec``:
``query`` -> question, ``grading_notes`` -> reference answer,
``source_fiche`` -> expected source file(s). A CSV carries no extra metadata,
so ``DatasetItem.metadata`` stays empty.

``limit`` is intentionally not handled here: it is a run-level concern applied
after loading, exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .item import DatasetItem
from .registry import register_loader

QUERY_COL = "query"
REFERENCE_COL = "grading_notes"
SOURCE_COL = "source_fiche"


@register_loader(".csv")
def load_csv(path: Path) -> list[DatasetItem]:
    df = pd.read_csv(path)

    missing = {QUERY_COL, REFERENCE_COL} - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV dataset {path} is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}."
        )

    has_source = SOURCE_COL in df.columns
    items: list[DatasetItem] = []
    for _, row in df.iterrows():
        source = (
            str(row[SOURCE_COL])
            if has_source and pd.notna(row[SOURCE_COL])
            else ""
        )
        items.append(
            DatasetItem(
                query=str(row[QUERY_COL]),
                reference=str(row[REFERENCE_COL]),
                source_fiche=source,
            )
        )
    return items
