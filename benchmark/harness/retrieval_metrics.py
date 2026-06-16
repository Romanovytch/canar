"""
Deterministic retrieval metrics (no LLM, no RAGAS).

These score the *ranked list of retrieved sources* against the expected
relevant source(s) for a question — the retrieval-quality metrics required by
the project roadmap (Hit Rate@k, MRR, Recall@k) plus Precision@k and nDCG@k.
Latency is measured in the pipeline and carried on PipelineOutput.

Relevance is matched the same way as the old `retrieval_hit`: an expected
source counts as retrieved if its string appears in a retrieved path, or the
path ends with its file name. `expected` may list several relevant sources
separated by ";" (the end-June dataset will have multi-source questions).

Pure Python — only the stdlib `math`. No extra dependency.
"""

from __future__ import annotations

import math
from pathlib import Path


def make_is_relevant(expected: str):
    """Return (predicate, n_relevant) for a ';'-separated list of expected sources."""
    targets = [t.strip() for t in str(expected).split(";") if t.strip()]

    def is_relevant(path: str) -> bool:
        return any(t in path or path.endswith(Path(t).name) for t in targets)

    return is_relevant, len(targets)


def compute(ranked_paths: list[str], expected: str, k: int | None = None) -> dict[str, float]:
    """
    Compute retrieval metrics for one question.

    ranked_paths : retrieved source paths, best-first (CanaR's ranked output)
    expected     : relevant source(s), ';'-separated
    k            : cutoff for @k metrics; None = use everything retrieved
    """
    is_relevant, n_relevant = make_is_relevant(expected)
    k = k or len(ranked_paths)
    top_k = ranked_paths[:k]

    found = sum(1 for p in top_k if is_relevant(p))
    ranks = [i for i, p in enumerate(ranked_paths, 1) if is_relevant(p)]
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(top_k, 1) if is_relevant(p))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_relevant, k) + 1))

    return {
        "hit_rate": 1.0 if found else 0.0,            # Hit Rate@k
        "mrr": 1.0 / ranks[0] if ranks else 0.0,      # Mean Reciprocal Rank
        "recall": found / n_relevant if n_relevant else 0.0,   # Recall@k
        "precision": found / k if k else 0.0,         # Precision@k
        "ndcg": dcg / idcg if idcg else 0.0,          # nDCG@k
    }
