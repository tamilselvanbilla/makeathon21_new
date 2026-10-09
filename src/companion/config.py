"""Environment-based configuration with defaults tuned for a Raspberry Pi 4.

Every value can be overridden with an environment variable so models can be
swapped without code changes. When the device is detected as a Raspberry Pi,
cheaper speech-to-text settings are chosen because the Pi 4's four Cortex-A72
cores are shared by wake-word detection, STT, and the LLM.
"""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _detect_raspberry_pi() -> bool:
    try:
        model = Path("/proc/device-tree/model").read_text(errors="ignore")
    except OSError:
        return False
    return "raspberry pi" in model.casefold()


IS_RASPBERRY_PI = _detect_raspberry_pi()
CPU_COUNT = os.cpu_count() or 4


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    name: str
    path: Path
    chat_format: str
    context: int
    threads: int
    batch: int
    max_tokens: int
    temperature: float

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            name=os.getenv("LLM_MODEL", "Qwen3-0.6B-Q4_K_M"),
            path=Path(
                os.getenv(
                    "LLM_MODEL_PATH",
                    str(PROJECT_ROOT / "models" / "llm" / "Qwen3-0.6B-Q4_K_M.gguf"),
                )
            ),
            # Qwen uses ChatML. Set LLM_CHAT_FORMAT (e.g. "llama-3", "gemma")
            # when swapping in a different model family.
            chat_format=os.getenv("LLM_CHAT_FORMAT", "chatml"),
            # 2048 tokens of KV cache is ~230 MB for Qwen3-0.6B; fits a 4 GB Pi.
            context=_env_int("LLM_CONTEXT", 2048),
            threads=_env_int("LLM_THREADS", CPU_COUNT),
            batch=_env_int("LLM_BATCH", 256),
            # ~8-10 tokens/s on a Pi 4, so 128 tokens caps replies at ~15 s.
            max_tokens=_env_int("LLM_MAX_TOKENS", 128),
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
        )


@dataclass(frozen=True)
class STTConfig:
    model_size: str
    language: str
    beam_size: int
    threads: int
    hotwords: str

    @classmethod
    def from_env(cls) -> "STTConfig":
        return cls(
            # English-only models are faster and more accurate for English.
            model_size=os.getenv("WHISPER_MODEL_SIZE", "tiny.en" if IS_RASPBERRY_PI else "base.en"),
            language=os.getenv("WHISPER_LANGUAGE", "en"),
            # Greedy decoding is ~3x faster than beam 5 on a Pi 4.
            beam_size=_env_int("WHISPER_BEAM_SIZE", 1 if IS_RASPBERRY_PI else 5),
            threads=_env_int("WHISPER_THREADS", CPU_COUNT),
            # Words tiny.en tends to mishear; the wake name is added at startup.
            hotwords=os.getenv("WHISPER_HOTWORDS", "EMI PAN Aadhaar"),
        )


@dataclass(frozen=True)
class CaptureConfig:
    device: int | None
    sample_rate: int
    max_record_seconds: float
    max_wait_for_speech_seconds: float
    silence_seconds: float
    speech_rms_threshold: float

    @classmethod
    def from_env(cls) -> "CaptureConfig":
        device = os.getenv("MIC_DEVICE")
        return cls(
            device=int(device) if device else None,
            sample_rate=16_000,
            max_record_seconds=_env_float("MAX_RECORD_SECONDS", 15),
            max_wait_for_speech_seconds=_env_float("MAX_WAIT_FOR_SPEECH_SECONDS", 30),
            silence_seconds=_env_float("SILENCE_SECONDS", 0.8),
            speech_rms_threshold=_env_float("SPEECH_RMS_THRESHOLD", 450),
        )


@dataclass(frozen=True)
class WakeWordConfig:
    enabled: bool
    phrase: str
    conversation_timeout: float

    @classmethod
    def from_env(cls) -> "WakeWordConfig":
        return cls(
            enabled=_env_bool("WAKE_WORD", True),
            # Detected in Whisper transcripts; any phrase Whisper spells reliably.
            phrase=os.getenv("WAKE_PHRASE", "hey jarvis"),
            # After waking, follow-ups need no wake phrase until this much silence.
            conversation_timeout=_env_float("CONVERSATION_TIMEOUT", 30),
        )


@dataclass(frozen=True)
class MemoryConfig:
    path: Path | None
    retention_days: int
    follow_up_minutes: float
    top_k: int
    embeddings: bool
    embedding_model: str
    embedding_dir: Path
    min_similarity: float

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        return cls(
            # MEMORY=0 keeps memory in RAM for the session only; nothing is written.
            path=Path(os.getenv("MEMORY_FILE", str(PROJECT_ROOT / "data" / "memory.sqlite3")))
            if _env_bool("MEMORY", True)
            else None,
            # Older notes and conversations are deleted at startup.
            retention_days=_env_int("MEMORY_RETENTION_DAYS", 30),
            # "And my wife's?" uses the previous question if it was this recent.
            follow_up_minutes=_env_float("FOLLOW_UP_MINUTES", 10),
            top_k=_env_int("MEMORY_TOP_K", 3),
            # Semantic search finds paraphrases ("power tool" -> drill); see
            # scripts/eval_memory_retrieval.py for the measurements behind these defaults.
            embeddings=_env_bool("MEMORY_EMBEDDINGS", True),
            embedding_model=os.getenv("EMBEDDING_MODEL", "minilm-int8"),
            embedding_dir=Path(os.getenv("EMBEDDING_DIR", str(PROJECT_ROOT / "models" / "embedding"))),
            min_similarity=_env_float("MEMORY_MIN_SIMILARITY", 0.35),
        )


@dataclass(frozen=True)
class KnowledgeConfig:
    path: Path
    primary_user: str | None
    top_k: int
    currency: str

    @classmethod
    def from_env(cls) -> "KnowledgeConfig":
        return cls(
            path=Path(os.getenv("KNOWLEDGE_FILE", str(PROJECT_ROOT / "knowledge_base" / "personal_data.json"))),
            # Whose records "my"/"I" refer to; defaults to the most frequent owner.
            primary_user=os.getenv("PRIMARY_USER") or None,
            # Records sent to the LLM per question; each one adds prompt time on a Pi.
            top_k=_env_int("KNOWLEDGE_TOP_K", 4),
            currency=os.getenv("CURRENCY", "INR"),
        )


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    stt: STTConfig
    capture: CaptureConfig
    wake_word: WakeWordConfig
    knowledge: KnowledgeConfig
    memory: MemoryConfig
    online_lookups_enabled: bool
    default_place: str
    tts_enabled: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            llm=LLMConfig.from_env(),
            stt=STTConfig.from_env(),
            capture=CaptureConfig.from_env(),
            wake_word=WakeWordConfig.from_env(),
            knowledge=KnowledgeConfig.from_env(),
            memory=MemoryConfig.from_env(),
            online_lookups_enabled=_env_bool("ONLINE_LOOKUPS", True),
            # Place used for weather questions that don't name one.
            default_place=os.getenv("DEFAULT_PLACE", "Bengaluru"),
            tts_enabled=_env_bool("TTS_ENABLED", True),
        )
