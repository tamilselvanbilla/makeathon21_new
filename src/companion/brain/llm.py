"""Local GGUF language model, loaded once and shared by the whole pipeline."""

import re
from typing import TYPE_CHECKING

from ..config import LLMConfig
from ..telemetry import timed_event

if TYPE_CHECKING:
    from llama_cpp import Llama

# Qwen3 soft switch: skip the <think> reasoning trace, which would cost
# several seconds of generation on a Pi 4 before any answer appears.
NO_THINK = "/no_think"


class LocalLLM:
    """Chat-style wrapper around llama-cpp running entirely on this device."""

    def __init__(self, config: LLMConfig):
        if not config.path.is_file():
            raise FileNotFoundError(
                f"Local model not found at {config.path}. Download the GGUF file "
                "or set LLM_MODEL_PATH."
            )
        from llama_cpp import Llama

        self.config = config
        self._llm: Llama = Llama(
            model_path=str(config.path),
            chat_format=config.chat_format,
            n_ctx=config.context,
            n_threads=config.threads,
            n_batch=config.batch,
            verbose=False,
        )

    def chat(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """Return one cleaned assistant reply for a system and user message."""
        max_tokens = max_tokens or self.config.max_tokens
        with timed_event("llm_generation", model=self.config.name, max_tokens=max_tokens):
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{user}\n{NO_THINK}"},
                ],
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                top_p=0.95,
                repeat_penalty=1.1,
            )
        return clean_model_response(str(response["choices"][0]["message"]["content"] or ""))


def strip_thinking(response: str) -> str:
    """Drop Qwen3 <think> blocks, including one cut off by max_tokens."""
    response = re.sub(r"(?s)<think>.*?</think>", " ", response)
    return re.sub(r"(?s)<think>.*", " ", response)


def clean_model_response(response: str) -> str:
    """Remove thinking traces, labels, serialized records, and duplicate sentences."""
    cleaned = strip_thinking(response)
    cleaned = re.sub(
        r"(?is)\s*(?:\[(?:financial|medical|documents|history)\]\s*)?\{[^{}]*\}",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\b(?:assistant|answer)\s*[:\-]\s*", "", cleaned)
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
