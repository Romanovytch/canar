from __future__ import annotations

from canar.app.agents import r_helpdesk
from canar.app.retrieval.models import RetrievalHit


def test_r_helpdesk_builds_context_from_retrieval_hits():
    hits = [
        RetrievalHit(
            text="Utilise filter() pour garder des lignes.",
            collection="utilitr",
            score=0.9,
            score_norm=1.0,
            source="utilitr",
            source_url="https://example.test/filter",
            section="Filtrer",
        )
    ]

    messages, sources = r_helpdesk.build_messages("Comment filtrer ?", hits)

    assert "[S1] Filtrer" in messages[1]["content"]
    assert "Utilise filter() pour garder des lignes." in messages[1]["content"]
    assert sources == [
        {
            "label": "S1",
            "url": "https://example.test/filter",
            "section": "Filtrer",
            "collection": "utilitr",
        }
    ]
