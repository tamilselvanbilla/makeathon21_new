"""Minimal local voice assistant using faster-whisper and Ollama."""

import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.signal import resample_poly


WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
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
    segments, _ = model.transcribe(audio, beam_size=5, language="en")
    return " ".join(segment.text.strip() for segment in segments).strip()


def local_model(prompt: str) -> str:
    """Send a prompt to a locally running Ollama server."""
    body = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Cannot reach Ollama. Start Ollama and make sure the configured "
            f"model '{OLLAMA_MODEL}' is available."
        ) from exc
    return str(result.get("response", "")).strip()


def personal_database(_intent: str) -> str:
    """No personal-data store is configured yet; do not invent personal facts."""
    return "No personal database is configured."


def speak(text: str) -> None:
    """Print the reply and speak it when pyttsx3 is installed."""
    print(f"Assistant: {text}")
    try:
        import pyttsx3
    except ImportError:
        print("(Install pyttsx3 to enable spoken replies.)")
        return

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def main() -> None:
    print("Available microphones:")
    input_device = choose_input_device()
    print(f"Loading Whisper '{WHISPER_MODEL_SIZE}'...")
    whisper = WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )
    print("Ready. Say 'quit' or 'exit' to stop.")

    while True:
        audio, _ = record_until_silence(input_device)
        question = speech_to_text(audio, whisper)
        if not question:
            print("No speech detected; listening again.")
            continue

        print(f"You: {question}")
        if question.casefold().strip(" .!?") in {"quit", "exit", "stop"}:
            print("Goodbye.")
            break

        try:
            intent = local_model(
                "Extract the personal-information lookup intent from this "
                f"question in one short sentence. Question: {question}"
            )
            data = personal_database(intent)
            answer = local_model(
                "Answer the user's question concisely. Do not invent personal "
                "facts; say when requested information is unavailable.\n"
                f"Question: {question}\nPersonal data: {data}"
            )
            speak(answer or "I couldn't generate a response.")
        except RuntimeError as exc:
            print(f"Error: {exc}")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)