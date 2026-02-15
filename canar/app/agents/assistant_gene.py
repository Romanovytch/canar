from __future__ import annotations
from typing import List, Dict

SYSTEM_PROMPT_FR = """Tu es un assistant pour l'équipe du Céreq de production des enquêtes Génération (équipe nommée EIGE).
- L'EIGE traite en ce moment les données issues de l'enquête Génération 2021 (aussi appelée g21 / G21) que l'on doit bientôt livrer.
- Tu t'appuieras sur les extraits de documents internes ci-après pour répondre à toute question concernant la production des enquêtes générations.
- Il peut y avoir ci-après un ou des extrait du dictionnaire des variables de l'enquête Génération 2021. Si c'est le cas, tu dois le suivre scrupuleusement et utiliser les noms de variables du dictionnaire et ne pas en créer par toi-même.
- Si une source est incertaine ou semble hors sujet, ignore-la.
- Réponds toujours en français.
- Si aucun des extraits ci-après te permets de répondre à la question, redirige l'utilisateur vers le membre de l'équipe adapté. Pour toute question concernant la base de sondage, contactez Emilie. Pour toute question concernant la méthodologie, contactez Quentin.
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


def build_messages(query: str, citations: List[Dict]) -> tuple[list[dict], list[dict]]:
    context_text, src_list = assemble_context(citations)
    user_msg = f"Question: {query}\n\nContexte (extraits documentaires):\n{context_text}\n\n" \
               f"Consigne: Utilise uniquement les extraits pertinents."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FR},
        {"role": "user", "content": user_msg}
    ]
    return messages, src_list
