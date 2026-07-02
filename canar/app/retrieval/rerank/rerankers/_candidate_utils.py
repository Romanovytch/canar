from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, asdict, is_dataclass
from typing import Any


class RerankerCandidateMixin:
    """Shared candidate handling for reranker implementations."""

    # Check direct fields first, then nested payload/metadata fields.
    TEXT_KEYS = ("text", "content", "page_content", "document", "chunk")

    # Add only metadata that is already present and useful to relevance judgment.
    METADATA_TEXT_KEYS = ("title", "section", "source")

    def _candidate_text(self, candidate: Any) -> str:
        # Prefer direct text/content fields from retrieval results.
        document_text = self._first_text_value(candidate, self.TEXT_KEYS)

        # Fall back to nested payload/metadata text used by compatibility dicts.
        metadata = self._candidate_metadata(candidate)
        if not document_text and metadata:
            document_text = self._first_text_value(metadata, self.TEXT_KEYS)

        # Make missing text explicit instead of silently scoring an empty document.
        if not document_text:
            raise ValueError("candidate must contain a text/content field")

        # Start with lightweight metadata that can help the reranker disambiguate.
        parts: list[str] = []
        for key in self.METADATA_TEXT_KEYS:
            value = self._candidate_value(candidate, key)
            if value is None and metadata:
                value = metadata.get(key)
            if value:
                parts.append(f"{key.title()}: {value}")

        # Add the actual retrieved passage last so it remains the main signal.
        parts.append(f"Document: {document_text}")
        return "\n".join(parts)

    def _candidate_metadata(self, candidate: Any) -> Mapping[str, Any] | None:
        # RetrievalHit stores original Qdrant payload in metadata.
        metadata = self._candidate_value(candidate, "metadata")
        if isinstance(metadata, Mapping):
            return metadata

        # Compatibility dicts expose the same payload under payload.
        payload = self._candidate_value(candidate, "payload")
        if isinstance(payload, Mapping):
            return payload

        # No nested metadata was available.
        return None

    def _candidate_value(self, candidate: Any, key: str) -> Any:
        # Dict-like candidates are common in older retrieval wrappers.
        if isinstance(candidate, Mapping):
            return candidate.get(key)

        # Lightweight object candidates are used by the current retrieval dataclass.
        return getattr(candidate, key, None)

    def _first_text_value(self, candidate: Any, keys: tuple[str, ...]) -> str:
        # Walk accepted text field names in priority order.
        for key in keys:
            value = self._candidate_value(candidate, key)
            if value:
                return str(value)

        # Return an empty string when no supported text field exists.
        return ""

    def _with_rerank_score(self, candidate: Any, score: float) -> Any:
        # Dict candidates get a shallow copy with all original keys preserved.
        if isinstance(candidate, Mapping):
            enriched = dict(candidate)
            enriched["rerank_score"] = float(score)
            return enriched

        # Object candidates get a shallow copy so callers keep their original hits.
        enriched_candidate = copy.copy(candidate)

        try:
            enriched_candidate.rerank_score = float(score)
            return enriched_candidate
        except FrozenInstanceError:
            # Frozen dataclasses without slots can carry an extra attribute on the copy.
            try:
                object.__setattr__(enriched_candidate, "rerank_score", float(score))
                return enriched_candidate
            except (AttributeError, TypeError):
                return self._candidate_as_dict(candidate, float(score))
        except (AttributeError, TypeError):
            # Slot-only or immutable lightweight objects fall back to a field dict.
            return self._candidate_as_dict(candidate, float(score))

    def _candidate_as_dict(self, candidate: Any, score: float) -> dict[str, Any]:
        # Dataclasses can be converted without depending on project-specific types.
        if is_dataclass(candidate):
            enriched = asdict(candidate)
        # Plain objects usually expose their fields through __dict__.
        elif hasattr(candidate, "__dict__"):
            enriched = vars(candidate).copy()
        # Last-resort fallback keeps the original object available to the caller.
        else:
            enriched = {"candidate": candidate}

        # Attach the reranker score without altering existing retrieval score fields.
        enriched["rerank_score"] = score
        return enriched

    def _read_rerank_score(self, candidate: Any) -> float:
        # Sorting supports both dict and object results from _with_rerank_score.
        if isinstance(candidate, Mapping):
            return float(candidate["rerank_score"])
        return float(candidate.rerank_score)
