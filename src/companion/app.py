"""Builds the companion from configuration and runs it."""

import argparse
import time

from .brain.knowledge import KnowledgeBase, load_knowledge
from .brain.llm import LocalLLM
from .config import IS_RASPBERRY_PI, AppConfig
from .device.indicator import ConsoleIndicator
from .device.mute import SoftwareMuteSwitch
from .device.tts import PrintSpeaker, make_speaker
from .online_gateway import OnlineGateway
from .pipeline import Assistant, MicInput, TextInput
from .telemetry import log_event


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Private, offline AI companion.")
    parser.add_argument("--text", action="store_true", help="type requests instead of speaking")
    parser.add_argument("--no-tts", action="store_true", help="print replies instead of speaking")
    parser.add_argument("--offline", action="store_true", help="disable all online lookups")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = AppConfig.from_env()
    started_at = time.perf_counter()
    print(f"Device: {'Raspberry Pi' if IS_RASPBERRY_PI else 'development machine'}")

    indicator = ConsoleIndicator()
    print(f"Loading local model '{config.llm.name}' ({config.llm.threads} threads)...")
    llm = LocalLLM(config.llm)

    if args.text:
        source = TextInput()
    else:
        # Imported here so text mode works without audio libraries installed.
        from .audio.capture import MicrophoneCapture, choose_input_device
        from .audio.stt import Transcriber

        capture = MicrophoneCapture(config.capture, choose_input_device(config.capture.device))
        print(f"Microphone: {capture.device_name} at {capture.sample_rate} Hz")
        print(f"Loading Whisper '{config.stt.model_size}' (beam {config.stt.beam_size})...")
        source = MicInput(capture, Transcriber(config.stt), SoftwareMuteSwitch(), indicator)

    speaker = PrintSpeaker() if args.text or args.no_tts or not config.tts_enabled else make_speaker()
    assistant = Assistant(
        llm=llm,
        knowledge=KnowledgeBase(
            load_knowledge(config.knowledge.path),
            primary_user=config.knowledge.primary_user,
            top_k=config.knowledge.top_k,
            currency=config.knowledge.currency,
        ),
        gateway=OnlineGateway(enabled=config.online_lookups_enabled and not args.offline),
        indicator=indicator,
        speaker=speaker,
    )

    startup_ms = (time.perf_counter() - started_at) * 1000
    log_event("startup", duration_ms=startup_ms, raspberry_pi=IS_RASPBERRY_PI)
    print(f"Ready in {startup_ms / 1000:.1f}s. Say or type 'exit' to stop.")
    assistant.run(source)
