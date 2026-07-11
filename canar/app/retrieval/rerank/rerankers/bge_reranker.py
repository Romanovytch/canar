from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from canar.app.retrieval.models import RetrievalHit
from canar.app.retrieval.rerank.rerankers._candidate_utils import (
    candidate_text,
    read_rerank_score,
    resolve_reranker_device,
    with_rerank_score,
)


class BGEReranker:
    """Small BGE cross-encoder wrapper for already-retrieved candidates."""

    # Keep the default model focused on the requested BGE reranker benchmark target.
    DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        instruction: str | None = None,
        # higher max_length = more context per candidate, more memory/time
        # lower max_length = faster, less memory, but more truncation risk
        max_length: int = 8192,
        device: str | None = None,
        # auto uses CUDA when available; cuda/cpu force placement; None leaves defaults.
    ) -> None:
        # Store model settings without loading model weights during construction.
        self.model_name = model_name
        self.instruction = instruction
        self.max_length = max_length
        self.device = device

        # These objects are initialized lazily on the first rerank call.
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalHit],
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        """Return candidates sorted by BGE rerank score, highest first."""
        # Reject blank queries because pairwise scoring is meaningless without one.
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        # Reject negative limits early so callers get a clear configuration error.
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be greater than or equal to 0")

        # Empty retrieval output remains empty after reranking.
        if not candidates or top_k == 0:
            return []

        # Build one cross-encoder pair per candidate while preserving input order.
        documents = [candidate_text(candidate) for candidate in candidates]
        pairs = [(query, document) for document in documents]

        # Score every query-document pair with the local BGE sequence classifier.
        scores = self._score_pairs(pairs)

        # Attach rerank scores to shallow copies so retrieval scores stay untouched.
        scored_candidates = [
            with_rerank_score(candidate, score)
            for candidate, score in zip(candidates, scores, strict=True)
        ]

        # Sort by reranker score while keeping the original relative order for ties.
        ordered = sorted(
            enumerate(scored_candidates),
            key=lambda item: (read_rerank_score(item[1]), -item[0]),
            reverse=True,
        )
        reranked = [candidate for _, candidate in ordered]

        # Return all candidates unless the caller requested a smaller top_k.
        if top_k is None:
            return reranked
        return reranked[:top_k]

    def _ensure_model(self) -> None:
        # Avoid importing heavy ML libraries until reranking is actually used.
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BGEReranker requires torch and transformers. "
                "Install the project dependencies before using reranking."
            ) from exc

        # BGE rerankers use the standard tokenizer/model cross-encoder interface.
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load the sequence-classification head used by the BGE reranker model card.
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()

        # Resolve auto lazily so importing the app does not require torch.
        resolved_device = resolve_reranker_device(torch, self.device)
        if resolved_device is not None:
            model = model.to(resolved_device)

        # Store everything needed for subsequent rerank calls.
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        # Loading is lazy to keep the class cheap to construct in application setup.
        self._ensure_model()

        # Narrow optional attributes after lazy initialization.
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        # Split pairs explicitly for the standard tokenizer text/text_pair API.
        queries = [query for query, _ in pairs]
        documents = [document for _, document in pairs]

        # Tokenize query-document pairs in the format expected by HF cross-encoders.
        inputs = self._tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        # Send inputs to the same device as the loaded model.
        model_device = next(self._model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        # BGE reranker logits are relevance scores; higher scores rank first.
        with self._torch.no_grad():
            logits = self._model(**inputs, return_dict=True).logits
            return logits.view(-1).float().tolist()
