from __future__ import annotations

from dataclasses import replace
from typing import Any

from canar.app.retrieval.models import RetrievalHit


def resolve_reranker_device(torch_module: Any, device: str | None) -> str | None:
    """Resolve profile device settings without importing torch at app startup."""
    if device is None:
        return None
    if device.strip().lower() == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


METADATA_TEXT_KEYS = ("title", "section", "source")


def candidate_text(candidate: RetrievalHit) -> str:
    if not candidate.text:
        raise ValueError("candidate text must be non-empty")

    parts: list[str] = []
    for key in METADATA_TEXT_KEYS:
        value = getattr(candidate, key, None)
        if value is None:
            value = candidate.metadata.get(key)
        if value:
            parts.append(f"{key.title()}: {value}")

    parts.append(f"Document: {candidate.text}")
    return "\n".join(parts)


def with_rerank_score(candidate: RetrievalHit, score: float) -> RetrievalHit:
    return replace(candidate, rerank_score=float(score))


def read_rerank_score(candidate: RetrievalHit) -> float:
    if candidate.rerank_score is None:
        raise ValueError("candidate has not been reranked")
    return candidate.rerank_score
