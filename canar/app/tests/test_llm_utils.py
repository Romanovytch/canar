from __future__ import annotations

from canar.app.retrieval.models import RetrievalHit
from canar.app.utils.llm_utils import assemble_context, build_universal_messages


def test_assemble_context_uses_retrieval_hit_fields():
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

    context, sources = assemble_context(hits)

    assert "[S1] Filtrer" in context
    assert "Utilise filter() pour garder des lignes." in context
    assert sources == [
        {
            "label": "S1",
            "url": "https://example.test/filter",
            "section": "Filtrer",
            "collection": "utilitr",
        }
    ]


def test_assemble_context_prefers_generation_text():
    hits = [
        RetrievalHit(
            text="Résumé court retrouvé.",
            generation_text="Texte original plus complet pour la génération.",
            collection="utilitr_summary",
            score=0.9,
            score_norm=1.0,
            section="Résumé",
        )
    ]

    context, _ = assemble_context(hits)

    assert "Texte original plus complet pour la génération." in context
    assert "Résumé court retrouvé." not in context


def test_build_universal_messages_combines_yaml_prompt_file_and_rag_context():
    messages = build_universal_messages(
        system_prompt="Prompt chargé depuis YAML",
        user_question="Comment convertir ce programme ?",
        file_content="data exemple; run;",
        rag_context="[S1] Documentation\nExtrait utile",
    )

    assert messages[0] == {"role": "system", "content": "Prompt chargé depuis YAML"}
    user_message = messages[1]["content"]
    assert "Comment convertir ce programme ?" in user_message
    assert "--- FICHIER JOINT ---" in user_message
    assert "data exemple; run;" in user_message
    assert "--- CONTEXTE DOCUMENTAIRE ---" in user_message
    assert "[S1] Documentation" in user_message
    assert "Cite les sources" in user_message


def test_build_universal_messages_omits_absent_optional_blocks():
    messages = build_universal_messages(
        system_prompt="Prompt simple",
        user_question="Bonjour",
    )

    assert messages == [
        {"role": "system", "content": "Prompt simple"},
        {"role": "user", "content": "Demande utilisateur :\nBonjour\n"},
    ]
