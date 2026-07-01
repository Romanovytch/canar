from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from canar.app.retrieval.models import RetrievalHit


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalHit],
        top_n: int | None = None,
    ) -> list[RetrievalHit]:
        """Return candidates ordered by descending reranker score."""
