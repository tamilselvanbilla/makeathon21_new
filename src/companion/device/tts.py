"""Spoken output."""

import os
import shutil
import subprocess
from typing import Callable, Protocol

TTS_ENGINE = os.getenv("TTS_ENGINE", "auto")
TTS_RATE = int(os.getenv("TTS_RATE", "150"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
TTS_VOICE = os.getenv("TTS_VOICE", "")
# ALSA device for aplay, e.g. "plughw:CARD=Headphones" for the Pi 4's 3.5 mm jack.
AUDIO_OUTPUT_DEVICE = os.getenv("AUDIO_OUTPUT_DEVICE", "default")
SPEECH_TIMEOUT_SECONDS = 60


class Speaker(Protocol):
    def say(self, text: str) -> None: ...


class PrintSpeaker:
    """Text-only output for development or when TTS is disabled."""

    def say(self, text: str) -> None:
        print(f"Assistant: {text}")


class EspeakSpeaker:
    """espeak-ng renders a WAV in memory and aplay plays it through ALSA.

    espeak-ng's own playback (also used by pyttsx3) needs a running
    PulseAudio/PipeWire server and fails on Pi OS Lite or over SSH with
    "audio open error: Unknown error 524". aplay writes to ALSA directly.
    """

    def __init__(
        self,
        device: str = AUDIO_OUTPUT_DEVICE,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.device = device
        self._run = run

    def synth_command(self) -> list[str]:
        command = [
            "espeak-ng",
            "--stdin",
            "--stdout",
            "-s", str(TTS_RATE),
            "-a", str(round(max(0.0, min(TTS_VOLUME, 2.0)) * 100)),
        ]
        if TTS_VOICE:
            command += ["-v", TTS_VOICE]
        return command

    def play_command(self) -> list[str]:
        return ["aplay", "-q", "-D", self.device]

    def say(self, text: str) -> None:
        print(f"Assistant: {text}")
        try:
            # Text goes through stdin, so a reply starting with "-" is never an option.
            wav = self._run(
                self.synth_command(),
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=SPEECH_TIMEOUT_SECONDS,
                check=True,
            ).stdout
            self._run(
                self.play_command(),
                input=wav,
                capture_output=True,
                timeout=SPEECH_TIMEOUT_SECONDS,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
            print(f"(Speech output failed: {exc.cmd[0]}: {detail or exc}. Check AUDIO_OUTPUT_DEVICE.)")
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"(Speech output failed: {exc})")


class Pyttsx3Speaker:
    """Speaks with the system voice (used on macOS and Windows)."""

    def say(self, text: str) -> None:
        print(f"Assistant: {text}")
        try:
            import pyttsx3
        except ImportError:
            print("(Install pyttsx3 to enable spoken replies.)")
            return

        # A fresh engine per reply avoids pyttsx3 hanging on reused engines.
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        if TTS_VOICE:
            engine.setProperty("voice", TTS_VOICE)
        engine.say(text)
        engine.runAndWait()
        engine.stop()


def make_speaker(engine: str = TTS_ENGINE) -> Speaker:
    """Pick espeak-ng + aplay where available (Linux/Pi), else pyttsx3."""
    if engine == "espeak" or (
        engine == "auto" and shutil.which("espeak-ng") and shutil.which("aplay")
    ):
        return EspeakSpeaker()
    return Pyttsx3Speaker()


if __name__ == "__main__":
    # Usage: python -m companion.device.tts ["text"]  (run from src/)
    import sys

    speaker = make_speaker()
    print(f"Engine: {type(speaker).__name__}, device: {AUDIO_OUTPUT_DEVICE}")
    speaker.say(" ".join(sys.argv[1:]) or "Speaker test. One, two, three.")
