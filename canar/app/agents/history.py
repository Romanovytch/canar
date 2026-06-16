MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 4000

ROLE_LABELS = {
    "user": "Utilisateur",
    "assistant": "Assistant",
}


def _get_message_attr(message, attr: str, default=None):
    if isinstance(message, dict):
        return message.get(attr, default)
    return getattr(message, attr, default)


def format_history_messages(
    history_messages=None,
    max_messages: int = MAX_HISTORY_MESSAGES,
    max_chars: int = MAX_HISTORY_CHARS,
) -> str:
    """Format recent conversation history for inclusion in an LLM prompt.

    The function accepts either Message-like objects or dictionaries with
    `role` and `content` fields.
    """
    if not history_messages:
        return "Aucun historique récent."

    recent_messages = list(history_messages)[-max_messages:]

    formatted_parts = []

    for message in recent_messages:
        role = _get_message_attr(message, "role", "")
        content = (_get_message_attr(message, "content", "") or "").strip()

        if not content:
            continue

        label = ROLE_LABELS.get(role, role or "Message")
        formatted_parts.append(f"{label} : {content}")

    formatted_history = "\n\n".join(formatted_parts).strip()

    if not formatted_history:
        return "Aucun historique récent."

    if len(formatted_history) > max_chars:
        formatted_history = "…\n" + formatted_history[-max_chars:].lstrip()

    return formatted_history


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
        {
            "role": "user",
            "content": "Et comment faire la même chose avec data.table ?",
        },
    ]

    formatted_history = format_history_messages(fake_history)

    print("=== Historique formaté ===")
    print(formatted_history)