"""Prompt text for the local model. Kept short: every token costs ~0.1 s on a Pi 4."""

NOT_IN_RECORDS_ANSWER = "I couldn't find that in your personal records, so I won't guess."
NO_RECORDS = "No personal records matched."


def build_system_prompt(primary_user: str, currency: str = "INR") -> str:
    return (
        f"You are a private, offline personal assistant for {primary_user}. "
        f"'I', 'me' and 'my' in questions mean {primary_user}; speak to them as "
        "'you'. Each knowledge "
        "record says whose it is; never attribute one person's record to "
        "someone else. Use only the supplied knowledge and the question. Never "
        "invent medical, financial, document, or history information; if the "
        "knowledge does not contain the answer, say so. For medical or "
        "financial decisions, state uncertainty and recommend qualified "
        "professional guidance. Reply in at most three short spoken sentences "
        "of natural language. Never reproduce raw JSON, field labels, or "
        f"repeated sentences. Money amounts are in {currency}. "
        f"Example: Question: what is my salary? Answer: Your monthly salary is {currency} 50,000."
    )


def build_user_prompt(question: str, knowledge: str) -> str:
    """Combine the question with local knowledge."""
    return f"Question: {question}\nKnowledge: {knowledge or NO_RECORDS}"


def build_advice_prompt(question: str, online_facts: str, knowledge: str) -> str:
    """Ask for one sentence of advice about facts that were already read out."""
    return (
        f"Question: {question}\nOnline facts (already read out to the user): {online_facts}\n"
        f"Knowledge: {knowledge or NO_RECORDS}\n"
        "Reply with one short sentence of practical advice that answers the question. "
        "Do not repeat the numbers."
    )
