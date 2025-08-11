from __future__ import annotations
from typing import List, Dict

SYSTEM_PROMPT_FR = """Tu es un assistant R pour des statisticiens de l'Insee.
- Donne des réponses pratiques, idiomatiques (tidyverse).
- Quand tu cites du code, utilise des blocs ```r.
- Tu as accès à des extraits de documentation interne (Contexte). Appuie-toi dessus.
- Si une source est incertaine ou hors sujet, ignore-la.
- Termine par une section “Sources” avec [S1], [S2], ... en listant les URLs fournis.
- Réponds en français.
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
               f"Consigne: Utilise uniquement les extraits pertinents. Cite [S1], [S2] si utilisés."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FR},
        {"role": "user", "content": user_msg}
    ]
    return messages, src_list
