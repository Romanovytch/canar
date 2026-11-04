"""
The project-owned dataset item model.

Minimal here on purpose: Task 2 only needs a return type for the loader
abstraction. Task 5 finalises the fields and any helpers. Every loader maps its
file format onto this shape, so the benchmark loop never sees CSV/YAML
specifics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetItem:
    """One benchmark question.

    query        : the question text
    reference    : the reference answer (RAGAS grading target)
    source_fiche : expected source file(s), ';'-separated (retrieval metrics)
    metadata     : format-specific extras the CSV can't carry (filled by the
                   YAML loader): question_type, expected_terms,
                   expected_sections, answer_present, difficulty, id, ...
    """

    query: str
    reference: str
    source_fiche: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
