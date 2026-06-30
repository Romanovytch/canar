"""
YAML dataset loader — for the richer mini_client benchmark structure
(``benchmark_utilitr_v0.yaml``, ``benchmark_dicovarg21_v0.yaml``).

Core fields plug into the same benchmark loop as the CSV loader:

    question              -> DatasetItem.query
    expected_answer       -> DatasetItem.reference
    expected_source_files -> DatasetItem.source_fiche   (';'-joined, list-form
                                                          also kept in metadata)

Everything else a question carries (id, question_type, expected_sections,
expected_terms, answer_present, difficulty, retrieval_notes, ...) plus the
dataset-level fields (dataset_id, document_group) goes into
``DatasetItem.metadata`` — so the YAML-specific signal is preserved and made
available to the execution/reporting layer, not flattened to the CSV fields.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .item import DatasetItem
from .registry import register_loader

QUESTION_KEY = "question"
ANSWER_KEY = "expected_answer"
SOURCES_KEY = "expected_source_files"

# Dataset-level keys copied into every item's metadata.
_DATASET_LEVEL_KEYS = ("dataset_id", "document_group")


@register_loader(".yaml", ".yml")
def load_yaml(path: Path) -> list[DatasetItem]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError(
            f"YAML dataset {path} is empty or missing the top-level 'questions' list."
        )

    dataset_meta = {k: data[k] for k in _DATASET_LEVEL_KEYS if k in data}

    items: list[DatasetItem] = []
    for i, q in enumerate(data.get("questions") or []):
        if not isinstance(q, dict):
            raise ValueError(f"YAML dataset {path}: question #{i + 1} is not a mapping.")

        missing = {QUESTION_KEY, ANSWER_KEY} - set(q)
        if missing:
            ident = q.get("id", f"#{i + 1}")
            raise ValueError(
                f"YAML dataset {path}: question {ident} is missing "
                f"required key(s): {sorted(missing)}."
            )

        sources = q.get(SOURCES_KEY) or []
        source_fiche = ";".join(str(s) for s in sources)

        # Keep every non-core field as metadata, then add the dataset-level fields.
        metadata = {k: v for k, v in q.items() if k not in {QUESTION_KEY, ANSWER_KEY}}
        metadata.update(dataset_meta)

        items.append(
            DatasetItem(
                query=str(q[QUESTION_KEY]),
                reference=str(q[ANSWER_KEY]),
                source_fiche=source_fiche,
                metadata=metadata,
            )
        )
    return items
