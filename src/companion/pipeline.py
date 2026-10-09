"""One conversational turn: hear -> route -> (online lookup) -> reason locally -> speak."""

import re
import time
from dataclasses import replace
from datetime import timedelta
from typing import Protocol

import numpy as np

from .brain.knowledge import KnowledgeBase
from .brain.market import Portfolio
from .brain.memory import (
    ConversationMemory,
    MemoryCommand,
    describe,
    is_follow_up,
    is_worth_noting,
    note_answer,
    parse_memory_command,
    schedule_answer,
    second_person,
    that_clause,
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
from .tracing import Tracer, annotate, set_turn, span

MUTE_POLL_SECONDS = 0.1
# Content after "remind me": "call mom" becomes "I need to call mom"; "I have ..." stays.
STATEMENT_OR_TO = re.compile(r"^\W*(?:i|i'm|i've|we|my|our|to)\b", re.I)
# Reminders missed by more than this (device off) are not read out late.
LATE_REMINDER_LIMIT = timedelta(hours=12)
YES = re.compile(r"^\W*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|please(?: do)?|do it|go ahead|of course|correct|right|save it|remember it)\b", re.I)
NO = re.compile(r"^\W*(?:no|nope|nah|don'?t|do not|never ?mind|not needed|cancel)\b", re.I)
# The model promising to remember something it cannot save itself.
FALSE_PROMISE = re.compile(
    r"\b(?:I(?:'ll| will| shall) (?:remember|keep|note|remind|make a note|save)|keep (?:this|that|it) in mind|"
    r"(?:I'?ve|I have) (?:noted|saved|made a note|remembered)|(?:has|have) been (?:noted|saved))\b",
    re.I,
)


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
        with span("listen"):
            audio = self.capture.record_utterance(should_abort=self.mute.is_muted)
            annotate(audio_seconds=round(audio.size / 16000, 2))
        if not audio.size:
            self.indicator.show(IndicatorState.IDLE)
            return ""

        self.indicator.show(IndicatorState.THINKING)
        text = self.transcriber.transcribe(audio)
        if not text:
            set_turn(ignored="unclear audio")
        if text and len(re.findall(r"[A-Za-z]{2,}", text)) < 2:
            print(f"[LISTENING] ignored {text!r}: too short to be a request")
            set_turn(ignored="too short", chars=len(text))  # ignored speech is never stored
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
        tracer: Tracer | None = None,
    ):
        self.name = name
        self.tracer = tracer or Tracer()  # in RAM unless the app passes a stored one
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
        # What the next utterance completes: ("content", prefix) after a bare "remember
        # that", or ("confirm", statement) after "do you want me to remember that ...?".
        self._pending: tuple[str, str] | None = None

    def respond(self, text: str) -> str:
        """Answer one request. Reasoning always runs on the local model."""
        set_turn(input=text)
        pending, self._pending = self._pending, None
        if pending:
            reply = self._complete_pending(pending, text)
            if reply is not None:
                set_turn(route=f"pending_{pending[0]}")
                return reply
        command = parse_memory_command(text)
        if command:
            set_turn(route=f"memory_{command.action}")
            return self._memory_command(command)
        is_schedule, day = self.memory.schedule_question(text)
        if is_schedule:
            with span("schedule", day=str(day or "week")):
                items = self.memory.scheduled_on(day) if day else self.memory.upcoming()
                annotate(items=len(items))
            print(f"[MEMORY] {len(items)} scheduled item(s) for {day or 'the coming week'}")
            set_turn(route="schedule")
            return schedule_answer(items, day, self.memory.now().date())
        if is_worth_noting(text, self.memory.now()):
            self._pending = ("confirm", text)
            set_turn(route="offer_note")
            return f"Do you want me to remember{that_clause(text)}?"

        previous = self.memory.last_turn() if is_follow_up(text) else None
        if previous:
            set_turn(follow_up=True)
        # "And my wife's?" is searched and routed as "<previous question> and my wife's?".
        effective = f"{previous.question} {text}" if previous else text
        if is_news_question(text):
            set_turn(route="news")
            reply, keep = self._respond_with_news(text)
        elif self._is_market_question(effective):
            set_turn(route="market")
            reply, keep = self._respond_with_market(text, effective, previous)
        elif route(effective) is Intent.ONLINE_LOOKUP:
            set_turn(route="weather")
            reply, keep = self._respond_with_lookup(text, effective)
        else:
            reply, keep = self._respond_locally(text, effective, previous)
        set_turn(remembered=keep)
        if keep:  # honest "I don't know" replies are not remembered as facts
            self.memory.add_turn(text, reply)
        return reply

    def _respond_locally(self, text: str, effective: str, previous) -> tuple[str, bool]:
        self.indicator.show(IndicatorState.THINKING)
        with span("knowledge"):
            knowledge = self.knowledge.context_for(effective)
            # Which records (category and owner), never their contents.
            found = re.findall(r"^(\w+) record of ([^:]+):", knowledge, re.M)
            annotate(records=len(found), sources=[f"{c.lower()}/{o}" for c, o in found])
        with span("memory"):
            items = self.memory.search(text)
            annotate(items=len(items), kinds=[i.kind for i in items], exact=[i.exact for i in items])
        if items:
            print(f"[MEMORY] using {len(items)} remembered item(s)")
        if items and items[0].kind == "note" and (not knowledge or items[0].exact):
            # Answer from the user's own note directly; a small model may misquote it.
            set_turn(route="note" if items[0].exact else "note_hedged")
            return note_answer(items[0]), True
        remembered = describe(items)
        if not knowledge and not remembered and self.knowledge.is_personal(effective):
            # Nothing on record or in memory: answer honestly instead of guessing.
            set_turn(route="not_in_records")
            return NOT_IN_RECORDS_ANSWER, False
        set_turn(route="llm")
        earlier = f"{previous.question} You answered: {previous.answer}" if previous else ""
        if self.debug_context:
            print("[CONTEXT] records sent to the model:\n  " + (knowledge.replace("\n", "\n  ") or "(none)"))
            if remembered:
                print("[CONTEXT] memory sent to the model:\n  " + remembered.replace("\n", "\n  "))
        answer = require_local_answer(
            self.llm.chat(self.system_prompt, build_user_prompt(text, knowledge, remembered, earlier))
        )
        if EXAMPLE_FIGURE in answer and EXAMPLE_FIGURE not in f"{knowledge} {remembered} {earlier}":
            set_turn(guard="example_figure", model_answer=answer)
            return DIDNT_CATCH_ANSWER, False  # copied the prompt's example, not real data
        if FALSE_PROMISE.search(answer):
            set_turn(guard="false_promise", model_answer=answer)
            # Only the app saves notes; never let the model claim it did.
            print(f"[MEMORY] model claimed to remember: {answer!r}; asking instead")
            self._pending = ("confirm", text)
            return f"I haven't saved that. Do you want me to remember{that_clause(text)}?", False
        if is_uncertain(answer):
            set_turn(uncertain=True)
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
            set_turn(online_failed=str(exc), route="market_fallback")
            reply, keep = self._respond_locally(text, effective, previous)
            set_turn(route="market_fallback")
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
            set_turn(online_failed=str(exc))
            return f"I couldn't get the news. {exc} I won't guess.", False
        print(f"[ONLINE] fetched feeds={list(query.request.feeds)} (topic not sent)")
        self.indicator.show(IndicatorState.THINKING)
        if query.topic:
            print(f"[LOCAL] filtering {len(result.data)} headlines for the topic on-device")
        return headlines_reply(result, query), True

    def _complete_pending(self, pending: tuple[str, str], text: str) -> str | None:
        """The answer to our own question; None if the user moved on to something else."""
        kind, value = pending
        if kind == "confirm":
            if YES.match(text):
                return self._save_note(value)
            if NO.match(text):
                return "Okay, I won't remember it."
            return None
        if text.rstrip().endswith("?") or parse_memory_command(text):
            return None
        content = text.strip(" .!?")
        return self._save_note(f"I need to {content}" if value == "to" and not STATEMENT_OR_TO.match(content) else content)

    def _save_note(self, text: str) -> str:
        stored, when = self.memory.add_note(text)
        print(f"[MEMORY] note saved on this device: {stored!r}" + (f", reminder at {when.due}" if when and when.has_time else ""))
        reply = f"Okay, I'll remember{that_clause(stored)}."
        if when and when.has_time and when.due > self.memory.now():
            reply = reply[:-1] + ", and remind you then."
        return reply

    def _memory_command(self, command: MemoryCommand) -> str:
        if command.action == "remember":
            return self._save_note(command.text)
        if command.action == "remember_pending":
            self._pending = ("content", command.text)
            return "Sure. What should I remind you to do?" if command.text == "to" else "Sure. What should I remember?"
        if command.action == "forget_last":
            item = self.memory.forget_last()
            print("[MEMORY] last item deleted" if item else "[MEMORY] nothing to delete")
            if item is None:
                return "There's nothing to forget."
            if item.kind == "note":
                return f"Okay, I've forgotten that {second_person(item.question)}."
            return f"Okay, I've forgotten our last exchange about: {item.question.strip(' .!?')}."
        count = self.memory.forget_all()
        traces = self.tracer.forget_all()
        print(f"[MEMORY] all {count} item(s) and {traces} trace(s) deleted")
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
            set_turn(online_failed=str(exc))
            return f"I couldn't do that online lookup. {exc} I won't guess real-time information.", False
        print(f"[ONLINE] sent only place={request.place!r}, day={request.day!r} to {result.source}")
        print(f"[ONLINE] {result.source} returned: {result.text}")

        self.indicator.show(IndicatorState.THINKING)
        print("[LOCAL] reasoning about the result on-device")
        knowledge = self.knowledge.context_for(text)
        advice = self.llm.chat(self.system_prompt, build_advice_prompt(text, result.text, knowledge))
        reply = f"From {result.source}, online: {result.text}"
        return (reply if is_uncertain(advice) else f"{reply} {advice}"), True

    def _say(self, text: str) -> None:
        self.indicator.show(IndicatorState.SPEAKING)
        with span("speak", chars=len(text)):
            self.speaker.say(text)
        self.indicator.show(IndicatorState.IDLE)

    def due_announcement(self) -> str:
        """ "Reminder: you have a meeting with Jay at 7:00 AM." for reminders now due."""
        now = self.memory.now()
        items = [item for item in self.memory.due_reminders() if now - item.due <= LATE_REMINDER_LIMIT]
        if not items:
            return ""
        print(f"[MEMORY] {len(items)} reminder(s) due")
        return "Reminder: " + "; ".join(item.spoken() for item in items) + "."

    def briefing(self) -> str:
        """What is still ahead today, said once at startup."""
        now = self.memory.now()
        items = [item for item in self.memory.scheduled_on(now.date()) if not item.has_time or item.due > now]
        if not items:
            return ""
        return "Today you have: " + "; ".join(item.spoken() for item in items) + "."

    def run(self, source: InputSource, welcome: str | None = None) -> None:
        if welcome:
            self._say(welcome)
        for notice in (self.due_announcement(), self.briefing()):
            if notice:
                self._say(notice)
        self.indicator.show(IndicatorState.IDLE)
        while True:
            # Listening times out every MAX_WAIT_FOR_SPEECH_SECONDS, so reminders are
            # checked at least that often.
            notice = self.due_announcement()
            if notice:
                with self.tracer.turn():
                    set_turn(route="reminder", reply=notice)
                    self._say(notice)
            with self.tracer.turn() as trace:
                text = source.next_utterance()
                if not text:
                    # Keep a trace of speech that was heard but ignored; drop silent timeouts.
                    if trace is not None and (text is None or "ignored" not in trace.attrs):
                        trace.discard()
                    if text is None:
                        break
                    set_turn(route="ignored")
                    continue
                if route(text) is Intent.EXIT:
                    set_turn(route="exit", input=text)
                    self._say("Goodbye.")
                    break

                try:
                    reply = self.respond(text)
                except Exception as exc:  # An always-on device must survive one bad turn.
                    log_event("turn_failed", error=type(exc).__name__)
                    print(f"Error: {exc}")
                    if trace is not None:
                        trace.status = "error"
                    set_turn(error=type(exc).__name__)
                    reply = "Sorry, something went wrong on my side. Please try again."

                set_turn(reply=reply)
                self._say(reply)

