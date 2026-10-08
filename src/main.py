"""Minimal local voice assistant using Faster-Whisper and local Qwen GGUF."""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from llama_cpp import Llama
from scipy.signal import resample_poly

from assistant_log import timed_event
from control import ConsoleListenerState, PhysicalMuteSwitch
from knowledge import find_relevant_records, load_knowledge
from model_config import MODEL_CTX, MODEL_NAME, MODEL_PATH, MODEL_THREADS
from privacy import can_send_audio, classify_request
from reasoning_policy import require_local_answer
from tts import speak


WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
TARGET_SAMPLE_RATE = 16_000
MAX_RECORD_SECONDS = 15
MAX_WAIT_FOR_SPEECH_SECONDS = 30
SILENCE_SECONDS = 0.8
SPEECH_RMS_THRESHOLD = 450


def choose_input_device() -> int:
    """Show microphones and select the configured or default input device."""
    inputs = [
        (index, info)
        for index, info in enumerate(sd.query_devices())
        if info["max_input_channels"] > 0
    ]
    if not inputs:
        raise RuntimeError("No microphone input devices were found.")

    default_device = sd.default.device[0]
    configured_device = os.getenv("MIC_DEVICE")
    for index, info in inputs:
        marker = " (default)" if index == default_device else ""
        print(f"  {index}: {info['name']}{marker}")

    if configured_device:
        selected = int(configured_device)
    else:
        fallback = default_device if default_device is not None and default_device >= 0 else inputs[0][0]
        choice = input(f"Microphone index [Enter for {fallback}]: ").strip()
        selected = fallback if not choice else int(choice)

    if selected not in {index for index, _ in inputs}:
        raise RuntimeError(f"Device {selected} is not an available microphone.")
    return selected


def record_until_silence(device: int) -> tuple[np.ndarray, int]:
    """Record mono audio until speech is followed by silence."""
    device_info = sd.query_devices(device, "input")
    sample_rate = int(device_info["default_samplerate"])
    block_size = max(1, int(sample_rate * 0.1))
    silence_blocks_needed = max(1, int(SILENCE_SECONDS / 0.1))
    chunks: list[np.ndarray] = []
    silent_blocks = 0
    speech_detected = False
    started_at = time.monotonic()

    print(f"Listening on {device_info['name']}... (Ctrl+C to quit)")
    with sd.InputStream(
        device=device,
        channels=1,
        samplerate=sample_rate,
        dtype="int16",
        blocksize=block_size,
    ) as stream:
        while True:
            chunk, overflowed = stream.read(block_size)
            if overflowed:
                print("Warning: microphone input overflowed; audio may be incomplete.")

            chunk = chunk.reshape(-1)
            chunks.append(chunk.copy())
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

            if rms >= SPEECH_RMS_THRESHOLD:
                speech_detected = True
                silent_blocks = 0
            elif speech_detected:
                silent_blocks += 1
                if silent_blocks >= silence_blocks_needed:
                    break

            elapsed = time.monotonic() - started_at
            if speech_detected and elapsed >= MAX_RECORD_SECONDS:
                break
            if not speech_detected and elapsed >= MAX_WAIT_FOR_SPEECH_SECONDS:
                return np.empty(0, dtype=np.float32), TARGET_SAMPLE_RATE

    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
    return audio, TARGET_SAMPLE_RATE


def speech_to_text(audio: np.ndarray, model: WhisperModel) -> str:
    """Transcribe a normalized mono waveform with Whisper."""
    if audio.size == 0:
        return ""
    segments, _ = model.transcribe(
        audio,
        beam_size=5,
        language="en",
        vad_filter=True,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def local_model(prompt: str, model: Llama) -> str:
    """Generate a response from the configured local model."""
    with timed_event(
        "llm_generation",
        model=MODEL_NAME,
        max_tokens=150,
    ):
        response = model(
            prompt,
            max_tokens=150,
            temperature=0.2,
            top_p=0.95,
            stop=["User:", "Assistant:", "Answer:"],
        )
    return clean_model_response(str(response["choices"][0]["text"]).strip())


def clean_model_response(response: str) -> str:
    """Remove labels, serialized records, and duplicate sentences."""
    cleaned = re.sub(
        r"(?is)\s*(?:\[(?:financial|medical|documents|history)\]\s*)?\{[^{}]*\}",
        " ",
        response,
    )
    cleaned = re.sub(
        r"(?i)\b(?:assistant|answer)\s*[:\-]\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip()
    ]

    unique_sentences: list[str] = []
    seen_sentences: set[str] = set()
    for sentence in sentences:
        normalized = sentence.casefold()
        if normalized in seen_sentences:
            continue
        seen_sentences.add(normalized)
        unique_sentences.append(sentence)

    return " ".join(unique_sentences).strip()


def personal_database(question: str) -> str:
    """Return configured JSON records relevant to the user's request."""
    return find_relevant_records(question, load_knowledge())


def main() -> None:
    print("Available microphones:")
    input_device = choose_input_device()
    mute_switch = PhysicalMuteSwitch()
    listener_state = ConsoleListenerState()
    print(f"Loading local model '{MODEL_NAME}' from '{MODEL_PATH}'...")
    model = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=MODEL_CTX,
        n_threads=MODEL_THREADS,
        verbose=False,
    )
    print(f"Loading Whisper '{WHISPER_MODEL_SIZE}'...")
    whisper = WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )
    print("Ready. Listening for speech. Mute with the physical switch.")

    while True:
        if mute_switch.is_muted():
            listener_state.set_listening(False)
            print("Listening disabled by mute switch.")
            continue

        listener_state.set_listening(True)
        audio, _ = record_until_silence(input_device)
        listener_state.set_listening(False)
        if not audio.size:
            continue

        question = speech_to_text(audio, whisper)
        if not question:
            continue

        print(f"You: {question}")
        if question.casefold().strip(" .!?") in {"quit", "exit", "stop"}:
            print("Goodbye.")
            break

        try:
            request_type = classify_request(question)
            data = personal_database(question)
            answer = local_model(
                "You are a private personal assistant acting as a financial "
                "adviser, medical adviser, personal document maintainer, "
                "and professional history maintainer. Use only the supplied "
                "knowledge and the user's question. Never invent medical, "
                "financial, document, or history information. For medical or "
                "financial decisions, state uncertainty and recommend qualified "
                "professional guidance. Return only one concise answer, explain "
                "the relevant facts in natural language, and never reproduce "
                "raw JSON or category-prefixed records. Do not repeat sentences "
                "or labels such as Answer:.\n"
                f"Request type: {request_type}\n"
                f"Question: {question}\nKnowledge: {data}",
                model,
            )
            answer = require_local_answer(answer)
            if not can_send_audio(True):
                speak(answer or "I couldn't generate a response.")
        except (RuntimeError, KeyError, IndexError) as exc:
            print(f"Error: {exc}")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)