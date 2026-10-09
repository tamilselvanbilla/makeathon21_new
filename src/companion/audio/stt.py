"""Local speech-to-text with faster-whisper (int8 on CPU)."""

import re
import sys

import numpy as np

from ..config import STTConfig
from ..telemetry import timed_event


def reliable(segment) -> bool:
    """Whisper's own quality signals (the thresholds Whisper uses itself): drop a
    segment that is probably silence it guessed at, or degenerate repetitive text."""
    if segment.no_speech_prob > 0.6 and segment.avg_logprob < -1.0:
        return False
    return segment.compression_ratio <= 2.4


def echoes_prompt(text: str, hotwords: str | None) -> bool:
    """On unclear audio Whisper repeats its prompt: with hotwords "Sam EMI PAN" it
    produced "Hello, my name is Sam EMI PAN". Two hotwords in a row mean an echo."""
    words = (hotwords or "").casefold().split()
    lowered = " ".join(re.findall(r"[a-z0-9']+", text.casefold()))
    return any(f" {a} {b} " in f" {lowered} " for a, b in zip(words, words[1:]))


def looks_like_noise(text: str) -> bool:
    """Transcripts like "M.D. M.D. M.D. S.B. S.B. S.B." that Whisper invents from
    background noise: long but made of very few distinct words."""
    words = re.findall(r"[a-z0-9']+", text.casefold())
    return len(words) >= 6 and len(set(words)) / len(words) < 0.4


class Transcriber:
    def __init__(self, config: STTConfig, hotwords: str | None = None):
        """`hotwords` biases recognition towards app words (wake name, EMI, PAN),
        which tiny.en otherwise mishears ("EMI" -> "UI")."""
        from faster_whisper import WhisperModel

        self.config = config
        self.hotwords = hotwords
        try:
            # Cached models only: without this, faster-whisper contacts huggingface.co at
            # every start to check for updates. scripts/setup.sh downloads the model.
            self._model = WhisperModel(
                config.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=config.threads,
                local_files_only=True,
            )
        except Exception as exc:
            if type(exc).__name__ != "LocalEntryNotFoundError":
                raise
            raise FileNotFoundError(
                f"Whisper model '{config.model_size}' is not downloaded. Run scripts/setup.sh "
                f"(for another size: WHISPER_MODEL_SIZE={config.model_size} scripts/setup.sh --skip-system)."
            ) from None

    def transcribe(self, audio: "np.ndarray | str", use_hotwords: bool = True, hotwords: str | None = None) -> str:
        """Transcribe a 16 kHz float32 waveform or an audio file path. `hotwords`
        replaces the default vocabulary for one call; the wake check passes only the
        wake name, because a list of words gets echoed as speech on room noise."""
        if isinstance(audio, np.ndarray) and audio.size == 0:
            return ""
        with timed_event("stt", model=self.config.model_size):
            segments, _ = self._model.transcribe(
                audio,
                beam_size=self.config.beam_size,
                language=self.config.language,
                vad_filter=True,
                hotwords=(hotwords or self.hotwords) if use_hotwords else None,
            )
            text = " ".join(segment.text.strip() for segment in segments if reliable(segment)).strip()
        if looks_like_noise(text) or (use_hotwords and echoes_prompt(text, hotwords or self.hotwords)):
            print("[STT] unclear audio ignored")
            return ""
        return text


if __name__ == "__main__":
    # Usage: python -m companion.audio.stt [audio/test.wav]  (run from src/)
    from ..config import PROJECT_ROOT

    path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "audio" / "test.wav")
    config = STTConfig.from_env()
    print(f"Loading Whisper '{config.model_size}'...")
    print("Transcription:", Transcriber(config).transcribe(path))
