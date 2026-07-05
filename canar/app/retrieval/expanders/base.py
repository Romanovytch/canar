from __future__ import annotations

from typing import Protocol

from canar.app.retrieval.models import ParentChildRetrievalParams, RetrievalHit


class HitExpander(Protocol):
    def expand(
        self,
        hits: list[RetrievalHit],
        params: ParentChildRetrievalParams,
    ) -> list[RetrievalHit]: ...
