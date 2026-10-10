"""Builds the companion from configuration and runs it."""

import argparse
import time

from .brain.knowledge import KnowledgeBase, load_knowledge
from .brain.llm import LocalLLM
from .brain.market import Portfolio
from .brain.memory import ConversationMemory
from .brain.prompts import build_welcome
from .config import IS_RASPBERRY_PI, AppConfig
from .device.indicator import ConsoleIndicator
from .device.mute import SoftwareMuteSwitch
from .device.tts import PrintSpeaker, make_speaker
from .online_gateway import OnlineGateway
from .pipeline import Assistant, MicInput, TextInput
from .telemetry import log_event
from .tracing import Tracer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Private, offline AI companion.")
    parser.add_argument("--text", action="store_true", help="type requests instead of speaking")
    parser.add_argument("--no-tts", action="store_true", help="print replies instead of speaking")
    parser.add_argument("--offline", action="store_true", help="disable all online lookups")
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

    microphone_name = None
    if args.text:
        source = TextInput()
    else:
        # Imported here so text mode works without audio libraries installed.
        from .audio.capture import MicrophoneCapture, choose_input_device
        from .audio.stt import Transcriber

        capture = MicrophoneCapture(config.capture, choose_input_device(config.capture.device))
        microphone_name = capture.device_name
        print(f"Microphone: {capture.device_name} at {capture.sample_rate} Hz")
        print(f"Loading Whisper '{config.stt.model_size}' (beam {config.stt.beam_size})...")
        hotwords = " ".join(filter(None, [config.stt.hotwords, config.default_place]))
        transcriber = Transcriber(config.stt, hotwords=hotwords)
        transcriber.warm_up()
        source = MicInput(capture, transcriber, SoftwareMuteSwitch(), indicator)

    if args.text or args.no_tts or not config.tts_enabled:
        speaker = PrintSpeaker()
    else:
        speaker = make_speaker(microphone_name=microphone_name)
        if getattr(speaker, "devices", None):
            print(f"Speaker: {speaker.device} (fallbacks: {', '.join(speaker.devices[1:]) or 'none'})")
    records = load_knowledge(config.knowledge.path)
    knowledge = KnowledgeBase(
        records,
        primary_user=config.knowledge.primary_user,
        top_k=config.knowledge.top_k,
        currency=config.knowledge.currency,
    )
    gateway = OnlineGateway(
        enabled=config.online_lookups_enabled and not args.offline,
        kinds=config.online_kinds,
        home_country=config.home_country,
    )
    print(f"Online lookups: {', '.join(sorted(gateway.kinds)) or 'off'}")
    traces = config.tracing
    tracer = Tracer(
        traces.path,
        enabled=traces.enabled,
        content=traces.content,
        console=traces.console,
        retention_days=traces.retention_days,
    )
    if traces.enabled:
        where = traces.path or "RAM only"
        print(f"Traces: {where} ({'with' if traces.content else 'without'} question text); view with scripts/traces.py")
    assistant = Assistant(
        llm=llm,
        name=config.assistant_name,
        debug_context=config.debug_context,
        tracer=tracer,
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

    print("Warming up the language model...")
    llm.warm_up(assistant.system_prompt)

    startup_ms = (time.perf_counter() - started_at) * 1000
    log_event("startup", duration_ms=startup_ms, raspberry_pi=IS_RASPBERRY_PI)
    hint = "Type 'exit' to stop." if args.text else "Speak your request. Ctrl+C stops."
    print(f"Ready in {startup_ms / 1000:.1f}s. {hint}")
    assistant.run(source, welcome=build_welcome(assistant.name, knowledge.primary_user))
