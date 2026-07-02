from __future__ import annotations

from canar.app.retrieval.rerank.rerankers.base import Reranker
from canar.app.retrieval.rerank.rerankers.bge_reranker import BGEReranker
from canar.app.retrieval.rerank.rerankers.qwen_reranker import QwenReranker


def build_reranker(
    name: str,
    *,
    device: str | None = None,
    max_length: int = 8192,
    model_name: str | None = None,
) -> Reranker:
    normalized_name = name.strip().lower()
    if normalized_name == "bge":
        return BGEReranker(
            model_name=model_name or BGEReranker.DEFAULT_MODEL_NAME,
            device=device,
            max_length=max_length,
        )
    if normalized_name == "qwen":
        return QwenReranker(
            model_name=model_name or QwenReranker.DEFAULT_MODEL_NAME,
            device=device,
            max_length=max_length,
        )
    raise ValueError(f"Unsupported reranker: {name!r}")
