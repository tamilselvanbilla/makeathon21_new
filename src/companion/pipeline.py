"""One conversational turn: hear -> route -> (online lookup) -> reason locally -> speak."""

import time
from typing import Protocol

import numpy as np

from .brain.knowledge import Records, find_relevant_records
from .brain.policy import require_local_answer
from .brain.prompts import SYSTEM_PROMPT, build_user_prompt
from .brain.router import Intent, route
from .device.indicator import Indicator, IndicatorState
from .device.mute import MuteSwitch
from .device.tts import Speaker
from .online_gateway import LookupUnavailable, OnlineGateway
from .telemetry import log_event

MUTE_POLL_SECONDS = 0.1


class ChatModel(Protocol):
    def chat(self, system: str, user: str) -> str: ...


class AudioCapture(Protocol):
    def record_utterance(self, should_abort) -> np.ndarray: ...


class SpeechToText(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class InputSource(Protocol):
    def next_utterance(self) -> str | None:
        """Return the next request, "" when nothing was heard, or None to stop."""
        ...


class TextInput:
    """Typed requests; for SSH sessions and development without a microphone."""

    def next_utterance(self) -> str | None:
        try:
            return input("You: ").strip()
        except EOFError:
            return None


class MicInput:
    """Spoken requests, gated by the mute switch and shown on the indicator."""

    def __init__(
        self,
        capture: AudioCapture,
        transcriber: SpeechToText,
        mute: MuteSwitch,
        indicator: Indicator,
    ):
        self.capture = capture
        self.transcriber = transcriber
        self.mute = mute
        self.indicator = indicator

    def next_utterance(self) -> str | None:
        if self.mute.is_muted():
            self.indicator.show(IndicatorState.MUTED)
            time.sleep(MUTE_POLL_SECONDS)
            return ""

        self.indicator.show(IndicatorState.LISTENING)
        audio = self.capture.record_utterance(should_abort=self.mute.is_muted)
        if not audio.size:
            self.indicator.show(IndicatorState.IDLE)
            return ""

        self.indicator.show(IndicatorState.THINKING)
        text = self.transcriber.transcribe(audio)
        if text:
            print(f"You: {text}")
        return text


class Assistant:
    def __init__(
        self,
        llm: ChatModel,
        records: Records,
        gateway: OnlineGateway,
        indicator: Indicator,
        speaker: Speaker,
    ):
        self.llm = llm
        self.records = records
        self.gateway = gateway
        self.indicator = indicator
        self.speaker = speaker

    def respond(self, text: str) -> str:
        """Answer one request. Reasoning always runs on the local model."""
        online_facts = None
        if route(text) is Intent.ONLINE_LOOKUP:
            self.indicator.show(IndicatorState.ONLINE)
            try:
                result = self.gateway.lookup(text)
            except LookupUnavailable as exc:
                return f"I couldn't do that online lookup. {exc} I won't guess real-time information."
            online_facts = f"{result.text} (source: {result.source})"

        self.indicator.show(IndicatorState.THINKING)
        knowledge = find_relevant_records(text, self.records)
        answer = self.llm.chat(SYSTEM_PROMPT, build_user_prompt(text, knowledge, online_facts))
        return require_local_answer(answer)

    def run(self, source: InputSource) -> None:
        self.indicator.show(IndicatorState.IDLE)
        while True:
            text = source.next_utterance()
            if text is None:
                break
            if not text:
                continue
            if route(text) is Intent.EXIT:
                self.speaker.say("Goodbye.")
                break

            try:
                reply = self.respond(text)
            except Exception as exc:  # An always-on device must survive one bad turn.
                log_event("turn_failed", error=type(exc).__name__)
                print(f"Error: {exc}")
                reply = "Sorry, something went wrong on my side. Please try again."

            self.indicator.show(IndicatorState.SPEAKING)
            self.speaker.say(reply)
            self.indicator.show(IndicatorState.IDLE)
