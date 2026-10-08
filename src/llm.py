from llama_cpp import Llama

MODEL_PATH = "models/llm/Qwen3-0.6B-Q4_K_M.gguf"

print("Loading local LLM...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

print("LLM loaded.")


def ask_llm(question):

    prompt = f"""
You are a private offline personal AI assistant.

Answer the user's question clearly and concisely.

User:
{question}

Assistant:
"""

    response = llm(
        prompt,
        max_tokens=200,
        temperature=0.2,
        stop=["User:", "Assistant:"]
    )

    answer = response["choices"][0]["text"].strip()

    return answer


if __name__ == "__main__":

    while True:

        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = ask_llm(question)

        print("\nAssistant:", answer)