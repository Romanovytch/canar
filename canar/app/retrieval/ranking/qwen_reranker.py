from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from canar.app.retrieval.ranking._candidate_utils import RerankerCandidateMixin


class QwenReranker(RerankerCandidateMixin):
    """Small Qwen3 reranker wrapper for already-retrieved candidates."""

    # Keep the default task instruction aligned with the Qwen3 reranker model card.
    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    # Prefer the current Qwen3 reranker target requested for this module.
    DEFAULT_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        instruction: str | None = None,
        #higher max_length = more context per candidate, more memory/time
        #lower max_length = faster, less memory, but more truncation risk
        max_length: int = 8192,
        device: str | None = None,
        #cuda for gpu, cpu for cpu, or None to let torch decide
    ) -> None:
        # Store model settings without loading model weights during construction.
        self.model_name = model_name
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self.max_length = max_length
        self.device = device

        # These objects are initialized lazily on the first rerank call.
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._prefix_tokens: list[int] | None = None
        self._suffix_tokens: list[int] | None = None
        self._token_true_id: int | None = None
        self._token_false_id: int | None = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        top_n: int | None = None,
    ) -> list[Any]:
        """Return candidates sorted by Qwen rerank score, highest first."""
        # Reject blank queries because pairwise scoring is meaningless without one.
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        # Reject negative limits early so callers get a clear configuration error.
        if top_n is not None and top_n < 0:
            raise ValueError("top_n must be greater than or equal to 0")

        # Empty retrieval output remains empty after reranking.
        if not candidates or top_n == 0:
            return []

        # Build one Qwen input string per candidate while preserving input order.
        documents = [self._candidate_text(candidate) for candidate in candidates]
        pairs = [self._format_instruction(query=query, document=document) for document in documents]

        # Score every query-document pair with the local Qwen model.
        scores = self._score_pairs(pairs)

        # Attach rerank scores to shallow copies so retrieval scores stay untouched.
        scored_candidates = [
            self._with_rerank_score(candidate, score)
            for candidate, score in zip(candidates, scores, strict=True)
        ]

        # Sort by reranker score while keeping the original relative order for ties.
        ordered = sorted(
            enumerate(scored_candidates),
            key=lambda item: (self._read_rerank_score(item[1]), -item[0]),
            reverse=True,
        )
        reranked = [candidate for _, candidate in ordered]

        # Return all candidates unless the caller requested a smaller top_n.
        if top_n is None:
            return reranked
        return reranked[:top_n]

    def _ensure_model(self) -> None:
        # Avoid importing heavy ML libraries until reranking is actually used.
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "QwenReranker requires torch and transformers. "
                "Install the project dependencies before using reranking."
            ) from exc

        # Qwen3 reranker expects left padding for causal-LM scoring.
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")

        # Load the causal LM form used by the official Qwen3 reranker recipe.
        model = AutoModelForCausalLM.from_pretrained(self.model_name).eval()

        # Move the model only when the caller explicitly selected a device.
        if self.device is not None:
            model = model.to(self.device)

        # Cache token IDs used to compare the model's yes/no relevance logits.
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        token_false_id = tokenizer.convert_tokens_to_ids("no")

        # Prefix and suffix follow the Qwen3 reranker chat-style scoring prompt.
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the "
            'Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        # Store everything needed for subsequent rerank calls.
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._token_true_id = token_true_id
        self._token_false_id = token_false_id
        self._prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    def _score_pairs(self, pairs: list[str]) -> list[float]:
        # Loading is lazy to keep the class cheap to construct in application setup.
        self._ensure_model()

        # Narrow optional attributes after lazy initialization.
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._prefix_tokens is not None
        assert self._suffix_tokens is not None
        assert self._token_true_id is not None
        assert self._token_false_id is not None

        # Reserve room for Qwen's prompt wrapper around each query-document pair.
        pair_max_length = self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        if pair_max_length <= 0:
            raise ValueError("max_length is too small for the Qwen reranker prompt")

        # Tokenize raw pair text before adding the reranker-specific prompt tokens.
        inputs = self._tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=pair_max_length,
        )

        # Wrap each encoded pair in the Qwen reranker prompt.
        for index, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = self._prefix_tokens + input_ids + self._suffix_tokens

        # Pad all pairs together for a single forward pass.
        padded_inputs = self._tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        # Send inputs to the same device as the loaded model.
        model_device = next(self._model.parameters()).device
        padded_inputs = {key: value.to(model_device) for key, value in padded_inputs.items()}

        # Convert final-token yes/no logits into a probability-like relevance score.
        with self._torch.no_grad():
            batch_scores = self._model(**padded_inputs).logits[:, -1, :]
            true_scores = batch_scores[:, self._token_true_id]
            false_scores = batch_scores[:, self._token_false_id]
            yes_no_scores = self._torch.stack([false_scores, true_scores], dim=1)
            log_probs = self._torch.nn.functional.log_softmax(yes_no_scores, dim=1)
            return log_probs[:, 1].exp().tolist()

    def _format_instruction(self, query: str, document: str) -> str:
        # Use the instruction-aware format expected by Qwen3 reranker models.
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"
