"""
Deterministic retrieval metrics (no LLM, no RAGAS).

These score the *ranked list of retrieved sources* against the expected
relevant source(s) for a question — the retrieval-quality metrics required by
the project roadmap (Hit Rate@k, MRR, Recall@k) plus Precision@k and nDCG@k.
Latency is measured in the pipeline and carried on PipelineOutput.

An expected source counts as retrieved if its string appears in a retrieved
path, or the path ends with its file name. `expected` may list several relevant
sources separated by ";" (the end-June dataset will have multi-source questions).

Pure Python — only the stdlib `math`. No extra dependency. Computed by hand,
which is fine at this scale; if the benchmark grows, consider `ranx` for the
ranking metrics instead.
"""

from __future__ import annotations

import math
from pathlib import Path


def make_is_relevant(expected: str):
    """Return (predicate, n_relevant) for a ';'-separated list of expected sources."""
    targets = [t.strip() for t in str(expected).split(";") if t.strip()]

    def is_relevant(path: str) -> bool:
        # NOTE: substring/filename matching can yield false positives depending on
        # how `expected` is written. Fine for now; validate against the real
        # benchmark dataset once it lands.
        return any(t in path or path.endswith(Path(t).name) for t in targets)

    return is_relevant, len(targets)


def _unique_fiches(ranked_paths: list[str]) -> list[str]:
    """Collapse chunks of the same fiche to one entry, keeping the best rank.

    The pipeline returns one path per retrieved chunk, so a relevant fiche shows
    up several times. Scoring documents (not chunks) is what keeps recall, nDCG
    and precision in [0, 1] — otherwise a fiche retrieved as N chunks counts as
    N relevant hits and recall can exceed 1.

    Dedup on the *full* path: chunks of one fiche share the same file_path, so
    this collapses them, while keeping two same-named fiches in different folders
    distinct (deduping on the bare filename would drop one of them).
    """
    seen, unique = set(), []
    for path in ranked_paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def compute(ranked_paths: list[str], expected: str, k: int | None = None) -> dict[str, float]:
    """
    Compute retrieval metrics for one question.

    ranked_paths : retrieved source paths, best-first (CanaR's ranked output)
    expected     : relevant source(s), ';'-separated
    k            : cutoff for @k metrics; None = use everything retrieved
    """
    is_relevant, n_relevant = make_is_relevant(expected)
    ranked_paths = _unique_fiches(ranked_paths)   # score documents, not chunks
    k = k or len(ranked_paths)
    top_k = ranked_paths[:k]

    found = sum(1 for p in top_k if is_relevant(p))
    # Ranks within top-k, so MRR is @k like every other metric here (the first
    # relevant doc beyond k does not count).
    ranks = [i for i, p in enumerate(top_k, 1) if is_relevant(p)]
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(top_k, 1) if is_relevant(p))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_relevant, k) + 1))

    return {
        "hit_rate": 1.0 if found else 0.0,            # Hit Rate@k
        "mrr": 1.0 / ranks[0] if ranks else 0.0,      # MRR@k
        "recall": found / n_relevant if n_relevant else 0.0,   # Recall@k
        "precision": found / k if k else 0.0,         # Precision@k
        "ndcg": dcg / idcg if idcg else 0.0,          # nDCG@k
    }
