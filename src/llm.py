import re

from llama_cpp import Llama

from assistant_log import timed_event
from model_config import MODEL_CTX, MODEL_NAME, MODEL_PATH, MODEL_THREADS

print(f"Loading local LLM '{MODEL_NAME}'...")

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=MODEL_CTX,
    n_threads=MODEL_THREADS,
    verbose=False,
)

print("LLM loaded.")


def clean_model_response(response: str) -> str:
    """Remove labels and repeated sentences from a model response."""
    cleaned = re.sub(r"(?i)\b(?:assistant|answer)\s*[:\-]\s*", "", response)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip()
    ]

    unique_sentences: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = sentence.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_sentences.append(sentence)

    return " ".join(unique_sentences).strip()


def ask_llm(question: str) -> str:
    prompt = f"""
You are a private offline personal AI assistant and trusted personal support
system. Your responsibilities include being a financial adviser, medical
adviser, personal document maintainer, and professional history maintainer.

Use the user's supplied information and available knowledge. Never invent
personal facts, medical diagnoses, financial advice, or records. Clearly
identify uncertainty and recommend consulting a qualified professional for
high-stakes medical or financial decisions. Keep answers concise, useful, and
safe. Do not repeat sentences or labels such as Answer:. Return only one
concise answer.

User:
{question}

Assistant:
"""

    with timed_event(
        "llm_generation",
        model=MODEL_NAME,
        max_tokens=150,
    ):
        response = llm(
            prompt,
            max_tokens=150,
            temperature=0.2,
            top_p=0.95,
            stop=["User:", "Assistant:", "Answer:"],
        )

    answer = response["choices"][0]["text"].strip()
    return clean_model_response(answer)


if __name__ == "__main__":
    while True:
        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = ask_llm(question)
        print("\nAssistant:", answer)