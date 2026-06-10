from __future__ import annotations

from canar.app.retrieval.models import RetrievalHit

SYSTEM_PROMPT_FR = """
Tu es “CoachR”, un formateur R très pédagogue pour un public venant majoritairement de SAS et peu 
familier des langages de programmation.
Objectif: expliquer R clairement, donner des exemples courts, et guider vers de bonnes pratiques 
utilisées en statistiques (nettoyage, indicateurs, pondération, contrôles).

Règles de pédagogie:
1) Commence par une explication simple (2–5 phrases), puis un exemple minimal exécutable.
2) Fais systématiquement un parallèle SAS→R quand cela aide (DATA step vs dplyr, PROC SQL vs 
dplyr/dbplyr, formats vs factors/labels, macro vs fonctions).
3) Privilégie des exemples inspirés d’enquêtes (variables, modalités, indicatrices, 
pondération) sans inventer de données sensibles.
4) Si l’utilisateur est bloqué, pose 1–2 questions maximum, sinon propose une hypothèse 
et avance.
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


def assemble_context(citations: list[RetrievalHit]) -> tuple[str, list[dict]]:
    """
    Map citations to labels [S1].. and return (context_text, source_list_for_ui)
    """
    lines = []
    srcs = []
    for i, hit in enumerate(citations, 1):
        label = f"S{i}"
        lines.append(f"[{label}] {hit.section}\n{hit.text}\n")
        srcs.append(
            {
                "label": label,
                "url": hit.source_url,
                "section": hit.section,
                "collection": hit.collection,
            }
        )
    context = "\n---\n".join(lines)
    return context, srcs


def build_messages(query: str, citations: list[RetrievalHit]) -> tuple[list[dict], list[dict]]:
    context_text, src_list = assemble_context(citations)
    user_msg = (
        f"Question: {query}\n\nContexte (extraits documentaires):\n{context_text}\n\n"
        f"Consigne: Utilise uniquement les extraits pertinents. Cite [S1], [S2] si utilisés."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FR},
        {"role": "user", "content": user_msg},
    ]
    return messages, src_list
