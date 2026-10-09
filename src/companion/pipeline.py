"""One conversational turn: hear -> route -> (online lookup) -> reason locally -> speak."""

import time
from dataclasses import replace
from typing import Protocol

import numpy as np

from .brain.knowledge import KnowledgeBase
from .brain.memory import (
    ConversationMemory,
    MemoryCommand,
    describe,
    is_follow_up,
    note_answer,
    parse_memory_command,
    second_person,
)
from .brain.policy import LOCAL_FALLBACK_ANSWER, require_local_answer, should_fallback
from .brain.prompts import NOT_IN_RECORDS_ANSWER, build_advice_prompt, build_system_prompt, build_user_prompt
from .brain.router import Intent, extract_place, parse_lookup, route
from .brain.wake import WakePhrase, is_sleep_command
from .device.indicator import Indicator, IndicatorState
from .device.mute import MuteSwitch
from .device.tts import Speaker
from .online_gateway import LookupUnavailable, OnlineGateway
from .telemetry import log_event

MUTE_POLL_SECONDS = 0.1
SAMPLE_RATE = 16_000
# While asleep, only the start of each utterance is transcribed to look for the
# wake phrase, which keeps the Pi's CPU free while people talk nearby.
WAKE_CHECK_SECONDS = 3.0


class ChatModel(Protocol):
    def chat(self, system: str, user: str) -> str: ...


class AudioCapture(Protocol):
    def record_utterance(self, should_abort, max_wait_seconds: float | None = None) -> np.ndarray: ...


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
    """Spoken requests, gated by the mute switch and shown on the indicator.

    With a wake phrase, the device has two states:
    - asleep (IDLE): speech is transcribed and kept only if it starts with the
      wake phrase ("Hey Jarvis, what's my EMI?"); everything else is discarded.
    - awake (LISTENING): a conversation. Follow-up requests need no wake phrase.
      It goes back to sleep after `conversation_timeout` seconds without speech,
      on a sleep command ("that's all", "stop listening"), or when muted.
    Without a wake phrase, every utterance is a request.
    """

    def __init__(
        self,
        capture: AudioCapture,
        transcriber: SpeechToText,
        mute: MuteSwitch,
        indicator: Indicator,
        wake: WakePhrase | None = None,
        conversation_timeout: float = 30.0,
    ):
        self.capture = capture
        self.transcriber = transcriber
        self.mute = mute
        self.indicator = indicator
        self.wake = wake
        self.conversation_timeout = conversation_timeout
        self.awake = wake is None

    def _go_to_sleep(self, reason: str) -> None:
        if self.wake and self.awake:
            self.awake = False
            print(f"[SLEEP] {reason}; say '{self.wake.phrase}' to start again.")

    def next_utterance(self) -> str | None:
        if self.mute.is_muted():
            self._go_to_sleep("muted")
            self.indicator.show(IndicatorState.MUTED)
            time.sleep(MUTE_POLL_SECONDS)
            return ""

        in_conversation = self.awake and self.wake is not None
        self.indicator.show(IndicatorState.LISTENING if self.awake else IndicatorState.IDLE)
        audio = self.capture.record_utterance(
            should_abort=self.mute.is_muted,
            max_wait_seconds=self.conversation_timeout if in_conversation else None,
        )
        if not audio.size:
            if in_conversation and not self.mute.is_muted():
                self._go_to_sleep(f"no follow-up for {self.conversation_timeout:.0f} s")
            return ""

        self.indicator.show(IndicatorState.THINKING)
        if not self.awake:
            return self._request_after_wake_phrase(audio)

        text = self.transcriber.transcribe(audio)
        if not text:
            return ""
        print(f"You: {text}")
        if self.wake:
            # Saying the wake phrase again mid-conversation is fine.
            stripped = self.wake.strip(text)
            if stripped is not None:
                text = stripped
            if is_sleep_command(text):
                self._go_to_sleep("conversation ended")
                return ""
        return text

    def _request_after_wake_phrase(self, audio: np.ndarray) -> str:
        head = audio[: int(WAKE_CHECK_SECONDS * SAMPLE_RATE)]
        request = self.wake.strip(self.transcriber.transcribe(head))
        if request is None:
            print("[IDLE] speech ignored: no wake phrase (nothing kept)")
            return ""
        if audio.size > head.size:
            # The request continued past the checked part: transcribe all of it.
            full = self.wake.strip(self.transcriber.transcribe(audio))
            request = full if full is not None else request

        if is_sleep_command(request):
            return ""
        self.awake = True
        print(f"[WAKE] '{self.wake.phrase}' heard; listening (follow-ups need no wake phrase)")
        if request:
            print(f"You: {request}")
        return request


class Assistant:
    def __init__(
        self,
        llm: ChatModel,
        knowledge: KnowledgeBase,
        gateway: OnlineGateway,
        indicator: Indicator,
        speaker: Speaker,
        default_place: str = "Bengaluru",
        memory: ConversationMemory | None = None,
    ):
        self.default_place = default_place
        self.memory = memory or ConversationMemory()
        self.llm = llm
        self.knowledge = knowledge
        self.system_prompt = build_system_prompt(knowledge.primary_user, knowledge.currency)
        self.gateway = gateway
        self.indicator = indicator
        self.speaker = speaker

    def respond(self, text: str) -> str:
        """Answer one request. Reasoning always runs on the local model."""
        command = parse_memory_command(text)
        if command:
            return self._memory_command(command)

        previous = self.memory.last_turn() if is_follow_up(text) else None
        # "And my wife's?" is searched and routed as "<previous question> and my wife's?".
        effective = f"{previous.question} {text}" if previous else text
        if route(effective) is Intent.ONLINE_LOOKUP:
            reply, keep = self._respond_with_lookup(text, effective)
        else:
            reply, keep = self._respond_locally(text, effective, previous)
        if keep:  # honest "I don't know" replies are not remembered as facts
            self.memory.add_turn(text, reply)
        return reply

    def _respond_locally(self, text: str, effective: str, previous) -> tuple[str, bool]:
        self.indicator.show(IndicatorState.THINKING)
        knowledge = self.knowledge.context_for(effective)
        items = self.memory.search(text)
        if items:
            print(f"[MEMORY] using {len(items)} remembered item(s)")
        if not knowledge and items and items[0].kind == "note":
            # Answer from the user's own note directly; a small model may misquote it.
            return note_answer(items[0]), True
        remembered = describe(items)
        if not knowledge and not remembered and self.knowledge.is_personal(effective):
            # Nothing on record or in memory: answer honestly instead of guessing.
            return NOT_IN_RECORDS_ANSWER, False
        earlier = f"{previous.question} You answered: {previous.answer}" if previous else ""
        answer = require_local_answer(
            self.llm.chat(self.system_prompt, build_user_prompt(text, knowledge, remembered, earlier))
        )
        return answer, answer != LOCAL_FALLBACK_ANSWER

    def _memory_command(self, command: MemoryCommand) -> str:
        if command.action == "remember":
            self.memory.add_note(command.text)
            print("[MEMORY] note saved on this device")
            return f"Okay, I'll remember that {second_person(command.text)}."
        if command.action == "forget_last":
            item = self.memory.forget_last()
            print("[MEMORY] last item deleted" if item else "[MEMORY] nothing to delete")
            if item is None:
                return "There's nothing to forget."
            if item.kind == "note":
                return f"Okay, I've forgotten that {second_person(item.question)}."
            return f"Okay, I've forgotten our last exchange about: {item.question.strip(' .!?')}."
        count = self.memory.forget_all()
        print(f"[MEMORY] all {count} item(s) deleted")
        return f"Done. I've erased {count} memories."

    def _respond_with_lookup(self, text: str, effective: str) -> tuple[str, bool]:
        """Fetch real-time facts online, then reason about them locally.

        Only the structured request (kind, place, day) leaves the device. The
        reply reads the facts out with their source, then adds one sentence of
        on-device advice, so the online and local parts stay distinguishable.
        """
        request = parse_lookup(effective, self.default_place)  # parsed locally
        place_now = extract_place(text)  # "what about in Chennai?" overrides the earlier place
        if place_now:
            request = replace(request, place=place_now)
        self.indicator.show(IndicatorState.ONLINE)
        try:
            result = self.gateway.lookup(request)
        except LookupUnavailable as exc:
            print(f"[ONLINE] {request.kind} lookup not performed: {exc}")
            return f"I couldn't do that online lookup. {exc} I won't guess real-time information.", False
        print(f"[ONLINE] sent only place={request.place!r}, day={request.day!r} to {result.source}")
        print(f"[ONLINE] {result.source} returned: {result.text}")

        self.indicator.show(IndicatorState.THINKING)
        print("[LOCAL] reasoning about the result on-device")
        knowledge = self.knowledge.context_for(text)
        advice = self.llm.chat(self.system_prompt, build_advice_prompt(text, result.text, knowledge))
        reply = f"From {result.source}, online: {result.text}"
        return (reply if should_fallback(advice) else f"{reply} {advice}"), True

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

