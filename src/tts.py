"""Text-to-speech helpers using pyttsx3."""

import os


TTS_RATE = int(os.getenv("TTS_RATE", "150"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
TTS_VOICE = os.getenv("TTS_VOICE", "")


def speak(text: str) -> None:
    """Print and speak text using the configured system voice."""
    print(f"Assistant: {text}")
    try:
        import pyttsx3
    except ImportError:
        print("(Install pyttsx3 to enable spoken replies.)")
        return

    engine = pyttsx3.init()
    engine.setProperty("rate", TTS_RATE)
    engine.setProperty("volume", TTS_VOLUME)
    if TTS_VOICE:
        engine.setProperty("voice", TTS_VOICE)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
