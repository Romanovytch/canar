from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from canar.app.retrieval.rerank.rerankers.base import Reranker
from canar.app.retrieval.rerank.rerankers.bge_reranker import BGEReranker
from canar.app.retrieval.rerank.rerankers.qwen_reranker import QwenReranker


@dataclass(frozen=True)
class RerankerSpec:
    model_name: str
    family: Literal["bge", "qwen"]


RERANKER_SPECS: dict[str, RerankerSpec] = {
    "bge-v2-m3": RerankerSpec("BAAI/bge-reranker-v2-m3", "bge"),
    "qwen-0.6b": RerankerSpec("Qwen/Qwen3-Reranker-0.6B", "qwen"),
    "qwen-4b": RerankerSpec("Qwen/Qwen3-Reranker-4B", "qwen"),
    "qwen-8b": RerankerSpec("Qwen/Qwen3-Reranker-8B", "qwen"),
}


def build_reranker(
    name: str,
    *,
    device: str | None = None,
    max_length: int = 8192,
) -> Reranker:
    normalized_name = name.strip().lower()
    spec = RERANKER_SPECS.get(normalized_name)
    if spec is None:
        raise ValueError(f"Unsupported reranker: {name!r}")

    if spec.family == "bge":
        return BGEReranker(
            model_name=spec.model_name,
            device=device,
            max_length=max_length,
        )

    if spec.family == "qwen":
        return QwenReranker(
            model_name=spec.model_name,
            device=device,
            max_length=max_length,
        )

    raise ValueError(f"Unsupported reranker family: {spec.family!r}")
