"""Prompt text for the local model. Kept short: every token costs ~0.1 s on a Pi 4."""

SYSTEM_PROMPT = (
    "You are a private, offline personal assistant that maintains the user's "
    "financial, medical, document, and professional-history records. Use only "
    "the supplied knowledge and the user's question. Never invent medical, "
    "financial, document, or history information. For medical or financial "
    "decisions, state uncertainty and recommend qualified professional "
    "guidance. Reply in at most three short spoken sentences of natural "
    "language. Never reproduce raw JSON, field labels, or repeated sentences."
)


def build_user_prompt(question: str, knowledge: str, online_facts: str | None = None) -> str:
    """Combine the question with local knowledge and any online lookup result."""
    parts = [f"Question: {question}", f"Knowledge: {knowledge}"]
    if online_facts:
        parts.append(f"Online facts: {online_facts}")
    return "\n".join(parts)
