"""Environment-based model configuration for easy model replacement."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = os.getenv("LLM_MODEL", "Qwen3-0.6B-Q4_K_M")
MODEL_PATH = Path(
    os.getenv(
        "LLM_MODEL_PATH",
        PROJECT_ROOT / "models" / "llm" / "Qwen3-0.6B-Q4_K_M.gguf",
    )
)
MODEL_CTX = int(os.getenv("LLM_CONTEXT", "2048"))
MODEL_THREADS = int(os.getenv("LLM_THREADS", "4"))
