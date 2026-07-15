def assemble_context(citations: list[dict]) -> tuple[str, list[dict]]:
    lines = []
    srcs = []
    for i, h in enumerate(citations, 1):
        p = h["payload"] or {}
        label = f"S{i}"
        url = p.get("url") or p.get("source_url")
        section = p.get("section") or ""
        text = p.get("text") or ""
        lines.append(f"[{label}] {section}\n{text}\n")
        srcs.append(
            {"label": label, "url": url, "section": section, "collection": h.get("collection")}
        )
    return "\n---\n".join(lines), srcs


def build_universal_messages(
    system_prompt: str,
    user_question: str,
    file_content: str | None = None,
    rag_context: str | None = None,
) -> list[dict]:
    """
    Assemble dynamiquement le prompt final en fonction des éléments disponibles.
    """
    # 1. On initialise le corps du message avec la question brute
    final_user_text = f"Demande utilisateur :\n{user_question}\n"

    # 2. Ajout conditionnel du fichier
    if file_content:
        final_user_text += f"\n--- FICHIER JOINT ---\n```\n{file_content}\n```\n"

    # 3. Ajout conditionnel du contexte Qdrant (Générique)
    if rag_context:
        final_user_text += (
            f"\n--- CONTEXTE DOCUMENTAIRE ---\n{rag_context}\n\n"
            f"Consigne : Réponds à la demande en t'appuyant sur le contexte ci-dessus. "
            f"Cite les sources sous la forme [S1], [S2] si tu les utilises."
        )

    # 4. Assemblage final au format OpenAI / ChatClient
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_user_text},
    ]
