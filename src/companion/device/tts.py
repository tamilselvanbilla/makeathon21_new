"""Spoken output."""

import os
from typing import Protocol

TTS_RATE = int(os.getenv("TTS_RATE", "150"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
TTS_VOICE = os.getenv("TTS_VOICE", "")


class Speaker(Protocol):
    def say(self, text: str) -> None: ...


class PrintSpeaker:
    """Text-only output for development or when TTS is disabled."""

    def say(self, text: str) -> None:
        print(f"Assistant: {text}")


class Pyttsx3Speaker:
    """Speaks with the system voice (espeak-ng on Raspberry Pi OS)."""

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
