from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from canar.app.retrieval.models import RetrievalHit


class FusionStrategy(Protocol):
    def fuse(
        self,
        ranked_lists: Sequence[Sequence[RetrievalHit]],
        top_k: int,
        weights: Sequence[float] | None = None,
    ) -> list[RetrievalHit]: ...
