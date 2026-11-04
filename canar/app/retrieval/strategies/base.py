from __future__ import annotations

from typing import Protocol

from canar.app.retrieval.models import RetrievalHit, RetrievalQuery


class RetrievalStrategy(Protocol):
    def search(self, query: RetrievalQuery) -> list[RetrievalHit]: ...
