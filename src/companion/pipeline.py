"""One conversational turn: hear -> route -> (online lookup) -> reason locally -> speak."""

import re
import time
from dataclasses import replace
from typing import Protocol

import numpy as np

from .brain.knowledge import KnowledgeBase
from .brain.market import Portfolio
from .brain.memory import (
    ConversationMemory,
    MemoryCommand,
    describe,
    is_follow_up,
    note_answer,
    parse_memory_command,
    second_person,
)
from .brain.policy import is_uncertain, require_local_answer
from .brain.news import headlines_reply, is_news_question, news_request
from .brain.prompts import (
    DIDNT_CATCH_ANSWER,
    EXAMPLE_FIGURE,
    NOT_IN_RECORDS_ANSWER,
    build_advice_prompt,
    build_system_prompt,
    build_user_prompt,
)
from .brain.router import Intent, extract_place, parse_lookup, route
from .device.indicator import Indicator, IndicatorState
from .device.mute import MuteSwitch
from .device.tts import Speaker
from .online_gateway import LookupUnavailable, OnlineGateway
from .telemetry import log_event

MUTE_POLL_SECONDS = 0.1


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
    Every utterance is treated as a request."""

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
        if text and len(re.findall(r"[A-Za-z]{2,}", text)) < 2:
            print(f"[LISTENING] ignored {text!r}: too short to be a request")
            return ""
        if text:
            print(f"You: {text}")
        return text


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
        portfolio: Portfolio | None = None,
        name: str = "Sam",
        debug_context: bool = False,
    ):
        self.name = name
        self.debug_context = debug_context
        self.portfolio = portfolio
        self.default_place = default_place
        self.memory = memory or ConversationMemory()
        self.llm = llm
        self.knowledge = knowledge
        self.system_prompt = build_system_prompt(knowledge.primary_user, knowledge.currency, name, knowledge.family)
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
        if is_news_question(text):
            reply, keep = self._respond_with_news(text)
        elif self._is_market_question(effective):
            reply, keep = self._respond_with_market(text, effective, previous)
        elif route(effective) is Intent.ONLINE_LOOKUP:
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
        if self.debug_context:
            print("[CONTEXT] records sent to the model:\n  " + (knowledge.replace("\n", "\n  ") or "(none)"))
            if remembered:
                print("[CONTEXT] memory sent to the model:\n  " + remembered.replace("\n", "\n  "))
        answer = require_local_answer(
            self.llm.chat(self.system_prompt, build_user_prompt(text, knowledge, remembered, earlier))
        )
        if EXAMPLE_FIGURE in answer and EXAMPLE_FIGURE not in f"{knowledge} {remembered} {earlier}":
            return DIDNT_CATCH_ANSWER, False  # copied the prompt's example, not real data
        return answer, not is_uncertain(answer)

    def _is_market_question(self, text: str) -> bool:
        # With market lookups switched off, these questions are answered from the records.
        return bool(self.portfolio) and "market" in self.gateway.kinds and self.portfolio.is_market_question(text)

    def _respond_with_market(self, text: str, effective: str, previous) -> tuple[str, bool]:
        """Live prices online (symbols only); holdings and all arithmetic on the device."""
        request, focus = self.portfolio.request(effective)
        self.indicator.show(IndicatorState.ONLINE)
        try:
            result = self.gateway.lookup(request)
        except LookupUnavailable as exc:
            print(f"[ONLINE] market lookup not performed: {exc}")
            reply, keep = self._respond_locally(text, effective, previous)
            return f"I couldn't get live prices, so this is from your saved records, which may be out of date. {reply}", keep
        print(f"[ONLINE] sent only symbols={list(request.symbols)} fund codes={list(request.fund_codes)}")
        self.indicator.show(IndicatorState.THINKING)
        print("[LOCAL] computing values and gains from your holdings on-device")
        return self.portfolio.answer(result, focus), True

    def _respond_with_news(self, text: str) -> tuple[str, bool]:
        """Whole feeds online; the topic is matched against headlines on the device."""
        query = news_request(text)
        self.indicator.show(IndicatorState.ONLINE)
        try:
            result = self.gateway.lookup(query.request)
        except LookupUnavailable as exc:
            print(f"[ONLINE] news lookup not performed: {exc}")
            return f"I couldn't get the news. {exc} I won't guess.", False
        print(f"[ONLINE] fetched feeds={list(query.request.feeds)} (topic not sent)")
        self.indicator.show(IndicatorState.THINKING)
        if query.topic:
            print(f"[LOCAL] filtering {len(result.data)} headlines for the topic on-device")
        return headlines_reply(result, query), True

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
        return (reply if is_uncertain(advice) else f"{reply} {advice}"), True

    def run(self, source: InputSource, welcome: str | None = None) -> None:
        if welcome:
            self.indicator.show(IndicatorState.SPEAKING)
            self.speaker.say(welcome)
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

