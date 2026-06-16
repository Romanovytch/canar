from __future__ import annotations
from typing import List, Dict
from canar.app.agents.history import format_history_messages

SYSTEM_PROMPT_FR = """
Tu es “CoachR”, un formateur R très pédagogue pour un public venant majoritairement de SAS et peu familier des langages de programmation.
Objectif: expliquer R clairement, donner des exemples courts, et guider vers de bonnes pratiques utilisées en production d’enquêtes (nettoyage, indicateurs, pondération, contrôles).

Règles de pédagogie:
1) Commence par une explication simple (2–5 phrases), puis un exemple minimal exécutable.
2) Fais systématiquement un parallèle SAS→R quand cela aide (DATA step vs dplyr, PROC SQL vs dplyr/dbplyr, formats vs factors/labels, macro vs fonctions).
3) Privilégie des exemples inspirés d’enquêtes (variables, modalités, indicatrices, pondération) sans inventer de données sensibles.
4) Si l’utilisateur est bloqué, pose 1–2 questions maximum, sinon propose une hypothèse et avance.
5) Toujours inclure:
   - “Exemple” (code R)
   - “À retenir” (3 bullets)
   - “Pièges fréquents (SAS→R)” (1–3 bullets) quand pertinent
6) Style:
   - Base R si nécessaire, sinon tidyverse pour la lisibilité.
   - Donne des noms d’objets parlants (df, individus, poids, etc.)
7) Ne donne pas d’infos non fondées; indique clairement ce qui est hypothèse.

Sortie: Markdown, blocs ```r```.
"""


def assemble_context(citations: List[Dict]) -> tuple[str, list[dict]]:
    """
    Map citations to labels [S1].. and return (context_text, source_list_for_ui)
    """
    lines = []
    srcs = []
    for i, h in enumerate(citations, 1):
        p = h["payload"] or {}
        label = f"S{i}"
        url = p.get("url") or p.get("source_url")
        section = p.get("section") or ""
        text = p.get("text") or ""
        lines.append(f"[{label}] {section}\n{text}\n")
        srcs.append({"label": label, "url": url, "section": section,
                    "collection": h.get("collection")})
    context = "\n---\n".join(lines)
    return context, srcs


def build_messages(query: str, citations: List[Dict], history_messages=None) -> tuple[list[dict], list[dict]]:
    context_text, src_list = assemble_context(citations)
    history = format_history_messages(history_messages)
    user_msg = (
        "Historique récent de la conversation :\n"
        f"{history}\n\n"
        "Question actuelle :\n"
        f"{query}\n\n"
        "Contexte (extraits documentaires) :\n"
        f"{context_text}\n\n"
        "Consigne : Utilise uniquement les extraits pertinents. "
        "Cite [S1], [S2] si utilisés."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FR},
        {"role": "user", "content": user_msg}
    ]
    return messages, src_list

if __name__ == "__main__":
    fake_history = [
        {
            "role": "user",
            "content": "Comment faire une jointure avec dplyr ?",
        },
        {
            "role": "assistant",
            "content": "Tu peux utiliser left_join(), inner_join() ou full_join().",
        },
    ]

    fake_citations = [
        {
            "payload": {
                "section": "Jointures avec dplyr",
                "text": "Le package dplyr propose plusieurs fonctions de jointure comme left_join(), inner_join(), right_join() et full_join().",
                "url": "https://example.org/dplyr-joins",
            },
            "collection": "utilitr_v1",
        }
    ]

    messages, src_list = build_messages(
        query="Et comment faire la même chose avec data.table ?",
        citations=fake_citations,
        history_messages=fake_history,
    )

    print("=== Message envoyé au LLM ===")
    print(messages[1]["content"])

    print("\n=== Sources UI ===")
    print(src_list)