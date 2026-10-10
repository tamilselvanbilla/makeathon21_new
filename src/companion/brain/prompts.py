"""Prompt text for the local model. Kept short: every token costs ~0.1 s on a Pi 4."""

import os

# The figure in the system prompt's example answer. If a reply contains it but the
# context doesn't, the model copied the example (seen on the Pi for the noise
# transcript "A": "Your monthly salary is INR 50,000").
EXAMPLE_FIGURE = "50,000"
DIDNT_CATCH_ANSWER = "Sorry, I didn't catch that. Could you say it again?"
NOT_IN_RECORDS_ANSWER = "No matched data found."
NO_NOTE_ANSWER = "I don't have a note about that, so I won't guess. Tell me \"remember that …\" and I'll keep it."
NO_RECORDS = "None needed or none found."
# Every answer is cut to this many words (see policy.limit_words). Not named in the
# prompt: with "50 words" in it, Qwen3-0.6B copied the example's 50,000 as a salary.
MAX_REPLY_WORDS = int(os.getenv("MAX_REPLY_WORDS", "50"))


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
        "records relevant to each question are given as Knowledge, and saved notes and "
        "earlier conversations as Memory. Treat Knowledge as authoritative for recorded facts. "
        "Prior assistant replies in Memory are conversation history, not verified facts; use them "
        "only to recall what was said, and prefer Knowledge when they conflict. Each record says "
        "whose it is; never attribute one person's record to someone else. For questions about "
        "the user or their family, use only "
        "Knowledge and Memory, never invent numbers, dates, or medical details, and if the answer "
        "is not there, say so. Answer general questions that are not about them briefly from "
        "general knowledge. Read out identity numbers only when asked for them. For medical or "
        "financial decisions, state uncertainty and recommend qualified professional guidance. "
        "Reply in at most three short spoken sentences of natural language. Never reproduce raw "
        f"JSON, field labels, or repeated sentences. Money amounts are in {currency}. "
        f"Example: Question: what is my salary? Answer: Your monthly salary is {currency} {EXAMPLE_FIGURE}."
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
        parts.append(f"Memory (saved notes and past conversations; prior assistant replies are not verified facts):\n{memory}")
    return "\n".join(parts)


def build_advice_prompt(question: str, online_facts: str, knowledge: str) -> str:
    """Ask for one sentence of advice about facts that were already read out."""
    return (
        f"Question: {question}\nOnline facts (already read out to the user): {online_facts}\n"
        f"Knowledge: {knowledge or NO_RECORDS}\n"
        "Reply with one short sentence of practical advice that answers the question. "
        "Do not repeat the numbers."
    )
