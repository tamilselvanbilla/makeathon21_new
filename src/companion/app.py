"""Builds the companion from configuration and runs it."""

import argparse
import time

from .brain.knowledge import KnowledgeBase, load_knowledge
from .brain.llm import LocalLLM
from .brain.market import Portfolio
from .brain.memory import ConversationMemory
from .brain.prompts import build_welcome
from .brain.wake import WakePhrase
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
    parser.add_argument("--no-wake-word", action="store_true", help="treat any speech as a request")
    return parser.parse_args(argv)


def load_embedder(config: AppConfig):
    """The memory embedding model, or None for keyword-only memory search."""
    if not config.memory.embeddings:
        return None
    from .brain.embeddings import Embedder

    try:
        embedder = Embedder(config.memory.embedding_dir, config.memory.embedding_model)
    except FileNotFoundError as exc:
        print(f"Memory search: keywords only ({exc})")
        return None
    print(f"Memory search: keywords + '{config.memory.embedding_model}' embeddings")
    return embedder


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
        via = f"arecord {capture.alsa_device}" if capture.backend == "arecord" else "PortAudio"
        print(f"Microphone: {capture.device_name} at {capture.sample_rate} Hz via {via}")
        print(f"Loading Whisper '{config.stt.model_size}' (beam {config.stt.beam_size})...")
        wake = None
        if config.wake_word.enabled and not args.no_wake_word:
            wake = WakePhrase(config.wake_word.phrase)
            print(
                f"Wake phrase: '{wake.phrase}' (conversation stays open "
                f"{config.wake_word.conversation_timeout:.0f} s after each reply)"
            )
        # Vocabulary for requests only; the wake check runs without hotwords (they get
        # echoed on noise), and the wake name is not needed after waking.
        hotwords = " ".join(filter(None, [config.stt.hotwords, config.default_place]))
        transcriber = Transcriber(config.stt, hotwords=hotwords)
        source = MicInput(
            capture,
            transcriber,
            SoftwareMuteSwitch(),
            indicator,
            wake,
            config.wake_word.conversation_timeout,
            debug=config.wake_word.debug,
        )

    speaker = PrintSpeaker() if args.text or args.no_tts or not config.tts_enabled else make_speaker()
    records = load_knowledge(config.knowledge.path)
    knowledge = KnowledgeBase(
        records,
        primary_user=config.knowledge.primary_user,
        top_k=config.knowledge.top_k,
        currency=config.knowledge.currency,
    )
    gateway = OnlineGateway(enabled=config.online_lookups_enabled and not args.offline, kinds=config.online_kinds)
    print(f"Online lookups: {', '.join(sorted(gateway.kinds)) or 'off'}")
    assistant = Assistant(
        llm=llm,
        name=WakePhrase(config.wake_word.phrase).name.title(),  # "hey sam" -> "Sam"
        knowledge=knowledge,
        gateway=gateway,
        portfolio=Portfolio.load(records, knowledge.primary_user, config.market_symbols_file),
        indicator=indicator,
        speaker=speaker,
        default_place=config.default_place,
        memory=ConversationMemory(
            config.memory.path,
            retention_days=config.memory.retention_days,
            follow_up_minutes=config.memory.follow_up_minutes,
            top_k=config.memory.top_k,
            embedder=load_embedder(config),
            min_similarity=config.memory.min_similarity,
        ),
    )

    startup_ms = (time.perf_counter() - started_at) * 1000
    log_event("startup", duration_ms=startup_ms, raspberry_pi=IS_RASPBERRY_PI)
    if args.text:
        hint = "Type 'exit' to stop."
    elif getattr(source, "wake", None):
        hint = f"Say '{source.wake.phrase}' and your request; 'that's all' ends a conversation. Ctrl+C stops."
    else:
        hint = "Speak your request. Ctrl+C stops."
    where = config.memory.path or "RAM only (MEMORY=0)"
    print(f"Memory: {where}, kept {config.memory.retention_days} days")
    print(f"Ready in {startup_ms / 1000:.1f}s. {hint}")
    wake = getattr(source, "wake", None)
    assistant.run(source, welcome=build_welcome(assistant.name, knowledge.primary_user, wake.phrase if wake else None))
