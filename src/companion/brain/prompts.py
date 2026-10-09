"""Prompt text for the local model. Kept short: every token costs ~0.1 s on a Pi 4."""

NOT_IN_RECORDS_ANSWER = "I couldn't find that in your personal records, so I won't guess."
NO_RECORDS = "None needed or none found."


def build_system_prompt(
    primary_user: str,
    currency: str = "INR",
    assistant_name: str = "Sam",
    family: list[str] | None = None,
) -> str:
    """`assistant_name` is used in the welcome, not here: naming a persona made the
    0.6B model speak as the user ("I am not overweight") in docs/benchmarks.md."""
    family_line = f" ({', '.join(family)})" if family else ""
    return (
        f"You are a private, offline personal assistant for {primary_user}. "
        f"'I', 'me' and 'my' in questions mean {primary_user}; speak to them as 'you'. "
        f"You hold the records of {primary_user} and their family{family_line}: finances "
        "(income, expenses, investments, loans), insurance policies, medical and health records, "
        "identity documents and other personal details, and work and education history. The "
        "records relevant to each question are given as Knowledge, and what the user told you "
        "earlier as Memory; trust them. Each record says whose it is; never attribute one "
        "person's record to someone else. For questions about the user or their family, use only "
        "Knowledge and Memory, never invent numbers, dates, or medical details, and if the answer "
        "is not there, say so. Answer general questions that are not about them briefly from "
        "general knowledge. Read out identity numbers only when asked for them. For medical or "
        "financial decisions, state uncertainty and recommend qualified professional guidance. "
        "Reply in at most three short spoken sentences of natural language. Never reproduce raw "
        f"JSON, field labels, or repeated sentences. Money amounts are in {currency}. "
        f"Example: Question: what is my salary? Answer: Your monthly salary is {currency} 50,000."
    )


def build_welcome(assistant_name: str, primary_user: str) -> str:
    """Spoken once at start-up."""
    return f"Hello {primary_user}, I'm {assistant_name}, your private assistant. Ask me anything."


def build_user_prompt(question: str, knowledge: str, memory: str = "", earlier: str = "") -> str:
    """Combine the question with local knowledge, memory, and the previous exchange."""
    parts = []
    if earlier:
        parts.append(f"Previous question: {earlier}")
    parts += [f"Question: {question}", f"Knowledge: {knowledge or NO_RECORDS}"]
    if memory:
        parts.append(f"Memory (things the user told you and past conversations):\n{memory}")
    return "\n".join(parts)


def build_advice_prompt(question: str, online_facts: str, knowledge: str) -> str:
    """Ask for one sentence of advice about facts that were already read out."""
    return (
        f"Question: {question}\nOnline facts (already read out to the user): {online_facts}\n"
        f"Knowledge: {knowledge or NO_RECORDS}\n"
        "Reply with one short sentence of practical advice that answers the question. "
        "Do not repeat the numbers."
    )
