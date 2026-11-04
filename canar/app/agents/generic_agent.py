from __future__ import annotations

from canar.app.retrieval.models import RetrievalHit

SYSTEM_PROMPT_FR = """You are a helpful assistant. Analyze the user's query and use the relevant 
document(s) provided below to answer it as accurately and helpfully as possible.

Answer the query using the information found in the document(s) above. 
If the document(s) do not contain enough information to answer, say so clearly 
instead of guessing.
"""


def assemble_context(citations: list[RetrievalHit]) -> tuple[str, list[dict]]:
    """
    Map citations to labels [S1].. and return (context_text, source_list_for_ui)
    """
    lines = []
    srcs = []
    for i, hit in enumerate(citations, 1):
        label = f"S{i}"
        lines.append(f"[{label}] {hit.section}\n{hit.generation_text or hit.text}\n")
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
