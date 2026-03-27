from __future__ import annotations


def format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return ""

    lines: list[str] = []
    for msg in history:
        role = msg.get("role", "").strip().lower()
        content = msg.get("content", "").strip()

        if not content:
            continue

        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    return "\n".join(lines)


def build_prompt(
    user_prompt: str,
    file_context: str = "",
    history: list[dict[str, str]] | None = None,
) -> str:
    parts: list[str] = []

    if file_context:
        parts.append("You have access to the following loaded file summaries.\n")
        parts.append(file_context.strip())

    if history:
        history_text = format_history(history)
        if history_text:
            parts.append("Conversation so far:\n")
            parts.append(history_text)

    parts.append("Latest user request:\n")
    parts.append(user_prompt.strip())

    return "\n\n".join(parts).strip()
