import ast
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from companion.brain import policy  # noqa: E402
from companion.brain.knowledge import KnowledgeBase  # noqa: E402
from companion.brain.memory import (  # noqa: E402
    ConversationMemory,
    MemoryCommand,
    note_answer,
    parse_memory_command,
)
from companion.brain.prompts import NOT_IN_RECORDS_ANSWER  # noqa: E402
from companion.brain.llm import clean_model_response  # noqa: E402
from companion.audio.capture import record_command  # noqa: E402
from companion.brain.router import Intent, extract_place, parse_lookup, route  # noqa: E402
from companion.config import CaptureConfig  # noqa: E402
from companion.device.indicator import ConsoleIndicator, IndicatorState  # noqa: E402
from companion.device.mute import SoftwareMuteSwitch  # noqa: E402
from companion.device.tts import EspeakSpeaker  # noqa: E402
from companion.brain.market import Portfolio  # noqa: E402
from companion.brain.news import news_request  # noqa: E402
from companion.online_gateway import (  # noqa: E402
    FORECAST_URL,
    GEOCODING_URL,
    LookupRequest,
    LookupUnavailable,
    OnlineGateway,
    check_host_allowed,
)
from companion.brain.wake import WakePhrase, is_sleep_command  # noqa: E402
from companion.pipeline import Assistant, MicInput  # noqa: E402

NETWORK_MODULES = {"requests", "httpx", "urllib3", "aiohttp", "socket", "http.client", "urllib.request"}


class FakeLLM:
    def __init__(self, reply="Your monthly income is 85000 rupees."):
        self.reply = reply
        self.calls = []

    def chat(self, system, user):
        self.calls.append(user)
        return self.reply


class RecordingIndicator:
    def __init__(self):
        self.states = []

    def show(self, state):
        self.states.append(state)


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


class FakeCapture:
    """Returns scripted utterances: a number of seconds of audio, or 0 for silence."""

    def __init__(self, *seconds):
        self.seconds = list(seconds) or [0.1]
        self.calls = 0
        self.waits = []

    def record_utterance(self, should_abort, max_wait_seconds=None):
        self.calls += 1
        self.waits.append(max_wait_seconds)
        duration = self.seconds.pop(0) if len(self.seconds) > 1 else self.seconds[0]
        return np.ones(int(16000 * duration), dtype=np.float32)


GEOCODE = {"results": [{"name": "Bengaluru", "country": "India", "latitude": 12.97, "longitude": 77.59}]}
FORECAST = {
    "current": {
        "temperature_2m": 27.2,
        "apparent_temperature": 28.4,
        "relative_humidity_2m": 51,
        "weather_code": 3,
        "wind_speed_10m": 9.0,
    },
    "daily": {
        "weather_code": [51, 61],
        "temperature_2m_max": [30.1, 29.0],
        "temperature_2m_min": [21.4, 20.0],
        "precipitation_probability_max": [30, 80],
    },
}


def chart(price, previous, name):
    return {"chart": {"result": [{"meta": {
        "shortName": name, "regularMarketPrice": price, "chartPreviousClose": previous, "currency": "INR"}}]}}


QUOTES = {
    "INFY.NS": chart(1023.4, 997.0, "INFOSYS LIMITED"),
    "%5ENSEI": chart(22520.45, 22603.1, "NIFTY 50"),
    "%5EBSESN": chart(72472.33, 72638.7, "S&P BSE SENSEX"),
    "TCS.NS": chart(2156.0, 2075.0, "TATA CONSULTANCY SERV LT"),
}
NAV = {"meta": {"scheme_name": "ICICI Prudential Balanced Advantage Fund - Direct Plan - Growth"},
       "data": [{"date": "08-10-2026", "nav": "84.40000"}]}
RSS = """<?xml version="1.0"?><rss><channel>
<item><title>TCS builds higher bench</title><pubDate>Fri, 09 Oct 2026 10:00:00 +0530</pubDate></item>
<item><title>Stock markets rebound led by TCS</title><pubDate>Fri, 09 Oct 2026 09:00:00 +0530</pubDate></item>
<item><title>Coal supplies rise 36%</title><pubDate>Fri, 09 Oct 2026 11:00:00 +0530</pubDate></item>
<item><title>Cricket betting racket busted</title><pubDate>Fri, 09 Oct 2026 08:00:00 +0530</pubDate></item>
</channel></rss>"""


class FakeFetch:
    """Stands in for the network: records what would have been sent."""

    def __init__(self, geocode=GEOCODE, fail=False):
        self.geocode = geocode
        self.fail = fail
        self.sent = []

    def __call__(self, url, params, as_text=False):
        check_host_allowed(url)
        self.sent.append((url, params))
        if self.fail:
            raise LookupUnavailable("I couldn't reach the service.")
        if "finance.yahoo" in url:
            symbol = url.rsplit("/", 1)[1]
            if symbol not in QUOTES:
                raise LookupUnavailable("unknown symbol")
            return QUOTES[symbol]
        if "mfapi" in url:
            return NAV
        if as_text:
            return RSS
        return self.geocode if url == GEOCODING_URL else FORECAST

    def sent_text(self):
        return " ".join(f"{url} {params}" for url, params in self.sent)


def frames(*levels):
    """80 ms frames with the given int16 amplitude (0 = silence)."""
    return [np.full(1280, level, dtype=np.int16) for level in levels]


CAPTURE = CaptureConfig(
    device=None,
    sample_rate=16000,
    max_record_seconds=15,
    max_wait_for_speech_seconds=30,
    silence_seconds=0.8,
    speech_rms_threshold=450,
)


class FakeTranscriber:
    """Returns scripted transcripts in order (or a fixed one) and records audio lengths."""

    def __init__(self, *texts):
        self.texts = list(texts) or ["what is my income"]
        self.lengths = []

    def transcribe(self, audio):
        self.lengths.append(len(audio) / 16000)
        return self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]


def conversation(transcripts, seconds=(1,), muted=False, timeout=30):
    capture = FakeCapture(*seconds)
    transcriber = FakeTranscriber(*transcripts)
    indicator = RecordingIndicator()
    source = MicInput(
        capture, transcriber, SoftwareMuteSwitch(muted), indicator, WakePhrase("hey jarvis"), timeout
    )
    return source, capture, transcriber, indicator


def quietly(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = fn(*args)
    return result, out.getvalue()


class ScriptedInput:
    def __init__(self, *lines):
        self.lines = list(lines)

    def next_utterance(self):
        return self.lines.pop(0) if self.lines else None


SAMPLE_RECORDS = {
    "financial": [
        {"owner": "John", "record_type": "income", "source": "Test sample", "monthly_income": 85000},
        {"owner": "John's wife", "record_type": "income", "source": "Test sample", "monthly_income": 62000},
        {
            "owner": "John",
            "record_type": "bank_loan",
            "source": "Test sample",
            "loan_type": "Home Loan",
            "monthly_installment": 9200,
        },
    ],
    "medical": [],
    "documents": [
        {"owner": "John's wife", "record_type": "family_member", "name": "Jane", "relationship": "Wife"},
        {"owner": "John", "record_type": "passport", "document_number": "TEST-PASSPORT-00001"},
    ],
    "history": [
        {"owner": "John's family", "record_type": "family_insurance", "provider": "LIC"},
        {"owner": "John", "record_type": "personal_accident_insurance", "valid_to": "2027-01-01"},
    ],
}


class Clock:
    def __init__(self, start=datetime(2026, 10, 9, 10, 0)):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, **delta):
        self.now += timedelta(**delta)


HOLDINGS = {
    "financial": [
        {"owner": "John", "record_type": "stock", "company": "Infosys Ltd.", "ticker": "INFY.NS",
         "quantity": 10, "purchase_price": 1750},
        {"owner": "John", "record_type": "mutual_fund", "scheme": "ICICI Prudential Balanced Advantage Fund",
         "scheme_code": 120377, "units": 323.5, "investment_amount": 25000},
        {"owner": "John's wife", "record_type": "stock", "company": "Wipro", "ticker": "WIPRO.NS",
         "quantity": 99, "purchase_price": 400},
    ]
}
SYMBOLS_FILE = ROOT / "knowledge_base" / "market_symbols.json"


def make_portfolio():
    return Portfolio.load(HOLDINGS, "John", SYMBOLS_FILE)


def make_assistant(llm=None, gateway=None, memory=None, portfolio=None):
    indicator = RecordingIndicator()
    speaker = FakeSpeaker()
    assistant = Assistant(
        llm=llm or FakeLLM(),
        knowledge=KnowledgeBase(SAMPLE_RECORDS),
        gateway=gateway or OnlineGateway(enabled=True, fetch=FakeFetch()),
        memory=memory,
        portfolio=portfolio,
        indicator=indicator,
        speaker=speaker,
    )
    return assistant, indicator, speaker


class PrivacyTests(unittest.TestCase):
    def test_lookup_classification(self):
        self.assertTrue(policy.is_allowed_cloud_lookup("weather in bengaluru"))
        self.assertFalse(policy.is_allowed_cloud_lookup("explain my financial data"))

    def test_gateway_only_accepts_structured_requests(self):
        gateway = OnlineGateway(fetch=FakeFetch())
        for payload in (b"\x00\x01raw-audio", np.zeros(16000, dtype=np.float32), "weather in bengaluru"):
            with self.assertRaises(TypeError):
                gateway.lookup(payload)

    def test_gateway_refuses_disabled_unsupported_or_sentence_like_requests(self):
        fetch = FakeFetch()
        cases = [
            OnlineGateway(enabled=False, fetch=fetch).lookup,
            OnlineGateway(fetch=fetch).lookup,
            OnlineGateway(fetch=fetch).lookup,
        ]
        requests = [
            LookupRequest("weather", "Bengaluru"),
            LookupRequest("news", "Bengaluru"),
            LookupRequest("weather", "my salary is 85000"),
        ]
        for lookup, request in zip(cases, requests):
            with self.assertRaises(LookupUnavailable):
                lookup(request)
        self.assertEqual(fetch.sent, [])

    def test_only_allowlisted_hosts(self):
        with self.assertRaises(LookupUnavailable):
            check_host_allowed("https://example.com/v1/search")
        check_host_allowed(GEOCODING_URL)
        check_host_allowed(FORECAST_URL)

    def test_weather_sends_only_place_and_coordinates(self):
        fetch = FakeFetch()
        result = OnlineGateway(fetch=fetch).lookup(LookupRequest("weather", "Bengaluru"))
        self.assertEqual(result.source, "Open-Meteo")
        self.assertIn("In Bengaluru, India it is 27 degrees and overcast", result.text)
        self.assertIn("30 percent chance of rain", result.text)
        (geo_url, geo_params), (_, forecast_params) = fetch.sent
        self.assertEqual(geo_params["name"], "Bengaluru")
        self.assertEqual((forecast_params["latitude"], forecast_params["longitude"]), (12.97, 77.59))

    def test_weather_tomorrow_and_unknown_place(self):
        tomorrow = OnlineGateway(fetch=FakeFetch()).lookup(LookupRequest("weather", "Bengaluru", "tomorrow"))
        self.assertIn("tomorrow: light rain, 20 to 29 degrees, with a 80 percent chance", tomorrow.text)
        with self.assertRaises(LookupUnavailable):
            OnlineGateway(fetch=FakeFetch(geocode={})).lookup(LookupRequest("weather", "Nowhere"))

    def test_only_gateway_imports_network_libraries(self):
        offenders = []
        for path in (SRC / "companion").rglob("*.py"):
            if path.name == "online_gateway.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                offenders += [f"{path.name}: {n}" for n in names if n in NETWORK_MODULES or n.split(".")[0] in NETWORK_MODULES]
        self.assertEqual(offenders, [])


class ReasoningTests(unittest.TestCase):
    def test_local_reasoning_has_fallback(self):
        self.assertTrue(policy.should_fallback("I cannot answer this accurately"))
        self.assertTrue(policy.should_fallback("   "))
        self.assertFalse(policy.should_fallback("Here is the answer"))

    def test_thinking_trace_and_duplicates_are_removed(self):
        raw = "<think>\nlet me reason\n</think>\n\nAnswer: It is sunny. It is sunny."
        self.assertEqual(clean_model_response(raw), "It is sunny.")
        self.assertEqual(clean_model_response("<think>never finished"), "")

    def test_lookup_parsing_keeps_only_place_and_day(self):
        self.assertEqual(
            parse_lookup("What's the weather in New York tomorrow?", "Bengaluru"),
            LookupRequest("weather", "New York", "tomorrow"),
        )
        self.assertEqual(parse_lookup("will it rain today", "Bengaluru"), LookupRequest("weather", "Bengaluru"))
        self.assertEqual(extract_place("weather in Mysore and should I go for a run?"), "Mysore")
        self.assertIsNone(extract_place("should I go for a run"))

    def test_rain_matches_whole_words_only(self):
        self.assertIs(route("will it rain in Chennai"), Intent.ONLINE_LOOKUP)
        self.assertIs(route("when is my train"), Intent.LOCAL_REASONING)
        self.assertIs(route("where is my umbrella?"), Intent.LOCAL_REASONING)
        self.assertIs(route("do I need an umbrella tomorrow?"), Intent.ONLINE_LOOKUP)

    def test_router(self):
        self.assertIs(route("Exit."), Intent.EXIT)
        self.assertIs(route("what's the weather today"), Intent.ONLINE_LOOKUP)
        self.assertIs(route("when does my passport expire"), Intent.LOCAL_REASONING)

    def test_knowledge_matches_are_readable_owner_labelled_prose(self):
        result = KnowledgeBase(SAMPLE_RECORDS).context_for("what is my monthly income")
        self.assertIn("record of John:", result)
        self.assertIn("Monthly Income: 85000", result)
        self.assertNotIn("{", result)


class KnowledgeSearchTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase(SAMPLE_RECORDS)

    def owners(self, question):
        return [match.owner for match in self.kb.search(question)]

    def test_primary_user_is_most_frequent_owner(self):
        self.assertEqual(self.kb.primary_user, "John")

    def test_my_means_primary_user_only(self):
        self.assertEqual(self.owners("what is my income"), ["John"])

    def test_naming_another_person_returns_their_records(self):
        self.assertEqual(set(self.owners("what is my wife's income")), {"John's wife"})
        self.assertIn("John's wife", self.owners("how much does Jane earn"))

    def test_family_questions_cover_everyone(self):
        self.assertIn("John's wife", self.owners("who is in my family"))

    def test_shared_family_records_are_included_by_default(self):
        self.assertIn("John's family", self.owners("do I have insurance"))

    def test_stemming_and_synonyms(self):
        self.assertIn("Home Loan", self.kb.context_for("what loans do I have"))
        self.assertIn("Home Loan", self.kb.context_for("what is my EMI"))

    def test_named_record_type_excludes_neighbouring_records(self):
        context = self.kb.context_for("what is my passport number")
        self.assertIn("TEST-PASSPORT-00001", context)
        self.assertNotIn("insurance", context)

    def test_missing_attribute_returns_nothing_instead_of_neighbours(self):
        self.assertEqual(self.kb.search("when does my passport expire"), [])
        self.assertTrue(self.kb.search("what is my passport number"))

    def test_synonym_note_explains_user_terms(self):
        self.assertIn("'emi' refers to installment", self.kb.context_for("what is my EMI"))

    def test_top_k_limits_results(self):
        self.assertEqual(len(KnowledgeBase(SAMPLE_RECORDS, top_k=1).search("my monthly income loan")), 1)

    def test_fts_operators_in_questions_are_harmless(self):
        self.assertEqual(self.kb.search('income" OR NOT * ('), self.kb.search("income"))

    def test_personal_question_detection(self):
        self.assertTrue(self.kb.is_personal("where did I park"))
        self.assertFalse(self.kb.is_personal("what is the capital of France"))


class PipelineTests(unittest.TestCase):
    def test_local_question_uses_local_model_and_knowledge(self):
        llm = FakeLLM()
        assistant, indicator, _ = make_assistant(llm=llm)
        reply = assistant.respond("what is my monthly income")
        self.assertEqual(reply, "Your monthly income is 85000 rupees.")
        self.assertIn("85000", llm.calls[0])
        self.assertIn(IndicatorState.THINKING, indicator.states)
        self.assertNotIn(IndicatorState.ONLINE, indicator.states)

    def test_unavailable_lookup_is_honest_and_skips_the_model(self):
        llm = FakeLLM()
        offline = OnlineGateway(fetch=FakeFetch(fail=True))
        assistant, indicator, _ = make_assistant(llm=llm, gateway=offline)
        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.respond("what is the weather in bengaluru")
        self.assertIn("won't guess", reply)
        self.assertEqual(llm.calls, [])
        self.assertIn(IndicatorState.ONLINE, indicator.states)

    def test_weather_reply_separates_online_facts_from_local_advice(self):
        llm = FakeLLM("Carry a light umbrella.")
        assistant, indicator, _ = make_assistant(llm=llm)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            reply = assistant.respond("Do I need an umbrella in Bengaluru?")
        self.assertTrue(reply.startswith("From Open-Meteo, online: In Bengaluru"))
        self.assertTrue(reply.endswith("Carry a light umbrella."))
        self.assertIn("Online facts (already read out", llm.calls[0])
        self.assertLess(indicator.states.index(IndicatorState.ONLINE), indicator.states.index(IndicatorState.THINKING))
        self.assertIn("[ONLINE] sent only place='Bengaluru', day='today' to Open-Meteo", out.getvalue())
        self.assertIn("[LOCAL]", out.getvalue())

    def test_uncertain_advice_is_dropped_but_facts_kept(self):
        assistant, _, _ = make_assistant(llm=FakeLLM("I can't say."))
        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.respond("weather in Bengaluru")
        self.assertTrue(reply.startswith("From Open-Meteo, online:"))
        self.assertNotIn("can't", reply)

    def test_personal_question_without_records_skips_the_model(self):
        llm = FakeLLM()
        assistant, _, _ = make_assistant(llm=llm)
        self.assertEqual(assistant.respond("what is my blood group"), NOT_IN_RECORDS_ANSWER)
        self.assertEqual(assistant.respond("when does my passport expire"), NOT_IN_RECORDS_ANSWER)
        self.assertEqual(llm.calls, [])

    def test_general_question_without_records_still_reaches_the_model(self):
        llm = FakeLLM("Paris.")
        assistant, _, _ = make_assistant(llm=llm)
        self.assertEqual(assistant.respond("what is the capital of France"), "Paris.")
        self.assertIn("No personal records matched.", llm.calls[0])

    def test_uncertain_model_answer_falls_back(self):
        assistant, _, _ = make_assistant(llm=FakeLLM("I do not know."))
        self.assertEqual(assistant.respond("what is my income"), policy.LOCAL_FALLBACK_ANSWER)

    def test_run_survives_errors_and_stops_on_exit(self):
        class BrokenLLM:
            def chat(self, system, user):
                raise RuntimeError("boom")

        assistant, indicator, speaker = make_assistant(llm=BrokenLLM())
        assistant.run(ScriptedInput("", "what is my income", "exit", "never reached"))
        self.assertEqual(len(speaker.said), 2)
        self.assertIn("something went wrong", speaker.said[0])
        self.assertEqual(speaker.said[1], "Goodbye.")
        self.assertIs(indicator.states[-1], IndicatorState.IDLE)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.memory = ConversationMemory(clock=self.clock)

    def test_memory_commands(self):
        self.assertEqual(
            parse_memory_command("Remember that I parked on level B2."),
            MemoryCommand("remember", "I parked on level B2"),
        )
        self.assertEqual(parse_memory_command("note my locker code is 4512"), MemoryCommand("remember", "my locker code is 4512"))
        self.assertEqual(parse_memory_command("Forget that."), MemoryCommand("forget_last"))
        self.assertEqual(parse_memory_command("forget everything"), MemoryCommand("forget_all"))
        self.assertIsNone(parse_memory_command("do you remember where I parked"))
        self.assertIsNone(parse_memory_command("what is my EMI"))

    def test_notes_are_found_by_stemmed_words(self):
        self.memory.add_note("I parked on level B2")
        self.assertIn("parked on level B2", self.memory.context_for("where did I park?"))

    def test_past_turns_only_for_recall_questions(self):
        self.memory.add_turn("what is my EMI", "Your EMI is INR 9200.")
        self.assertEqual(self.memory.context_for("what is my EMI"), "")
        self.assertIn("INR 9200", self.memory.context_for("what did you tell me about my EMI?"))

    def test_yesterday_filter(self):
        self.memory.add_turn("what is my EMI", "Your EMI is INR 9200.")
        self.clock.advance(days=1)
        self.memory.add_turn("when does my passport expire", "Not in your records.")
        context = self.memory.context_for("what did I ask yesterday?")
        self.assertIn("EMI", context)
        self.assertNotIn("passport", context)

    def test_follow_up_window(self):
        self.memory.add_turn("what is my monthly income", "Your monthly income is INR 85,000.")
        self.assertEqual(self.memory.last_turn().question, "what is my monthly income")
        self.clock.advance(minutes=11)
        self.assertIsNone(self.memory.last_turn())

    def test_forget(self):
        self.memory.add_note("my locker code is 4512")
        self.memory.add_note("I parked on level B2")
        self.assertEqual(self.memory.forget_last().question, "I parked on level B2")
        self.assertEqual(self.memory.forget_all(), 1)
        self.assertEqual(self.memory.context_for("locker code"), "")

    def test_persists_across_restarts_and_expires(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.sqlite3"
            ConversationMemory(path, clock=self.clock).add_note("I parked on level B2")
            self.assertIn("B2", ConversationMemory(path, clock=self.clock).context_for("where did I park"))
            self.clock.advance(days=31)
            self.assertEqual(ConversationMemory(path, clock=self.clock).context_for("where did I park"), "")


class ConceptEmbedder:
    """Deterministic stand-in for a sentence-embedding model: one dimension per concept."""

    CONCEPTS = [
        ("drill", "power tool"),
        ("umbrella", "brolly"),
        ("birthday", "wish"),
        ("parked", "park", "car", "level"),
        ("passport", "travel documents"),
    ]

    def __init__(self):
        self.calls = 0

    def embed(self, texts, query=False):
        self.calls += 1
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vec = np.array([1.0 if any(w in lowered for w in words) else 0.0 for words in self.CONCEPTS] + [0.1])
            vectors.append(vec / np.linalg.norm(vec))
        return np.array(vectors, dtype=np.float32)


class HybridMemoryTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.memory = ConversationMemory(clock=self.clock, embedder=ConceptEmbedder(), min_similarity=0.5)
        for note in ("I lent my drill to Ravi", "I left my umbrella in the car", "mom's birthday is on 12 March"):
            self.memory.add_note(note)

    def test_paraphrase_found_by_meaning_and_marked_inexact(self):
        hits = self.memory.search("who did I give my power tool to?")
        self.assertEqual(hits[0].question, "I lent my drill to Ravi")
        self.assertFalse(hits[0].exact)

    def test_full_keyword_match_is_exact_and_first(self):
        hits = self.memory.search("who has my drill?")
        self.assertEqual(hits[0].question, "I lent my drill to Ravi")
        self.assertTrue(hits[0].exact)

    def test_unrelated_question_finds_nothing(self):
        self.assertEqual(self.memory.search("what is my bank PIN?"), [])

    def test_near_miss_is_answered_with_a_hedge(self):
        hit = self.memory.search("when is my wife's birthday?")[0]
        self.assertFalse(hit.exact)
        self.assertTrue(note_answer(hit).startswith("I'm not certain, but the closest thing I remember is: on 9 October"))
        exact = self.memory.search("when is mom's birthday?")[0]
        self.assertEqual(note_answer(exact), "On 9 October you told me that mom's birthday is on 12 March.")

    def test_vectors_are_deleted_with_their_memories(self):
        self.memory.forget_last()
        self.assertEqual(self.memory._db.execute("SELECT count(*) FROM vectors").fetchone()[0], 2)
        self.memory.forget_all()
        self.assertEqual(self.memory._db.execute("SELECT count(*) FROM vectors").fetchone()[0], 0)

    def test_existing_memories_are_embedded_when_a_model_is_added(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.sqlite3"
            ConversationMemory(path, clock=self.clock).add_note("I lent my drill to Ravi")  # keyword-only
            upgraded = ConversationMemory(path, clock=self.clock, embedder=ConceptEmbedder(), min_similarity=0.5)
            self.assertEqual(upgraded.search("who has my power tool?")[0].question, "I lent my drill to Ravi")

    def test_keyword_only_mode_still_hedges_partial_matches(self):
        memory = ConversationMemory(clock=self.clock)
        memory.add_note("mom's birthday is on 12 March")
        self.assertFalse(memory.search("when is my wife's birthday?")[0].exact)
        self.assertTrue(memory.search("when is mom's birthday?")[0].exact)


class AssistantMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = ConversationMemory(clock=Clock())

    def test_remember_is_confirmed_without_the_model(self):
        llm = FakeLLM()
        assistant, _, _ = make_assistant(llm=llm, memory=self.memory)
        reply, _ = quietly(assistant.respond, "Remember that I parked on level B2")
        self.assertEqual(reply, "Okay, I'll remember that you parked on level B2.")
        self.assertEqual(llm.calls, [])

    def test_recall_answers_from_the_note_without_the_model(self):
        llm = FakeLLM()
        assistant, _, _ = make_assistant(llm=llm, memory=self.memory)
        quietly(assistant.respond, "remember that I parked on level B2")
        reply, out = quietly(assistant.respond, "Where did I park?")
        self.assertEqual(reply, "On 9 October you told me that you parked on level B2.")
        self.assertEqual(llm.calls, [])
        self.assertIn("[MEMORY] using 1 remembered item", out)

    def test_recall_of_past_answers_goes_through_the_model(self):
        llm = FakeLLM("I told you your EMI is INR 9200.")
        assistant, _, _ = make_assistant(llm=llm, memory=self.memory)
        quietly(assistant.respond, "what is my EMI")
        quietly(assistant.respond, "what did you tell me about my EMI?")
        self.assertIn("You answered: I told you", llm.calls[-1])

    def test_follow_up_uses_the_previous_question(self):
        llm = FakeLLM("Your wife's monthly income is INR 62,000.")
        assistant, _, _ = make_assistant(llm=llm, memory=self.memory)
        quietly(assistant.respond, "what is my monthly income")
        quietly(assistant.respond, "And my wife's?")
        prompt = llm.calls[-1]
        self.assertIn("Previous question: what is my monthly income", prompt)
        self.assertIn("record of John's wife", prompt)

    def test_honest_misses_are_not_remembered(self):
        assistant, _, _ = make_assistant(memory=self.memory)
        quietly(assistant.respond, "what is my blood group")
        self.assertIsNone(self.memory.last_turn())

    def test_weather_follow_up_changes_only_the_place(self):
        fetch = FakeFetch()
        assistant, _, _ = make_assistant(gateway=OnlineGateway(fetch=fetch), memory=self.memory)
        quietly(assistant.respond, "weather in Bengaluru tomorrow")
        quietly(assistant.respond, "What about in Mumbai?")
        geocoded = [params["name"] for url, params in fetch.sent if url == GEOCODING_URL]
        self.assertEqual(geocoded, ["Bengaluru", "Mumbai"])


class MarketTests(unittest.TestCase):
    def setUp(self):
        self.portfolio = make_portfolio()

    def ask(self, text, fetch=None):
        request, focus = self.portfolio.request(text)
        return self.portfolio.answer(OnlineGateway(fetch=fetch or FakeFetch()).lookup(request), focus)

    def test_market_questions_vs_local_ones(self):
        for text in ("how are my investments doing today?", "how is Infosys doing?", "what's the Nifty at?",
                     "TCS share price", "current value of my mutual fund"):
            self.assertTrue(self.portfolio.is_market_question(text), text)
        for text in ("what are my investments", "where did I work before infosys", "what is my EMI",
                     "what is my current role"):
            self.assertFalse(self.portfolio.is_market_question(text), text)

    def test_only_the_owners_holdings_and_whole_watchlist_are_requested(self):
        request, focus = self.portfolio.request("how is Infosys doing?")
        self.assertEqual(request.symbols, ("INFY.NS", "^NSEI", "^BSESN"))  # same watchlist whatever was asked
        self.assertEqual(request.fund_codes, ("120377",))
        self.assertNotIn("WIPRO.NS", request.symbols)  # someone else's holding
        self.assertEqual(focus, ["INFY.NS"])
        self.assertIn("TCS.NS", self.portfolio.request("TCS share price")[0].symbols)

    def test_holdings_never_leave_the_device(self):
        fetch = FakeFetch()
        self.ask("how are my investments doing?", fetch)
        sent = fetch.sent_text()
        for private in ("1750", "323.5", "25000", "10,234", "Infosys Ltd"):
            self.assertNotIn(private, sent)

    def test_values_and_gains_are_computed_locally(self):
        reply = self.ask("how are my investments doing today?")
        self.assertTrue(reply.startswith("From Yahoo Finance and AMFI, online: Nifty 50 is at 22,520.45, down 0.4 percent"))
        self.assertIn("Your 10 Infosys shares are worth 10,234 rupees, 7,266 rupees below what you paid, "
                      "down 41.5 percent.", reply)
        self.assertIn("323.5 units of ICICI Prudential Balanced Advantage Fund are worth 27,303 rupees, "
                      "2,303 rupees above", reply)
        self.assertIn("In total these are worth 37,537 rupees, 4,963 rupees below the 42,500 you invested", reply)
        self.assertTrue(reply.endswith("This is information, not investment advice."))

    def test_focus_on_funds_indices_or_a_named_company(self):
        fund = self.ask("what is the current value of my mutual fund?")
        self.assertIn("NAV is 84.40 rupees, as of 8 October", fund)
        self.assertNotIn("Infosys", fund)
        self.assertNotIn("Computed on this device", self.ask("what's the Nifty at?"))
        # The local name, not Yahoo's truncated "TATA CONSULTANCY SERV LT".
        self.assertIn("Tata Consultancy Services is at 2,156.00 rupees, up 3.9 percent", self.ask("TCS share price?"))

    def test_invalid_market_requests_are_refused_before_sending(self):
        fetch = FakeFetch()
        for request in (LookupRequest("market", symbols=("my salary",)), LookupRequest("market", fund_codes=("12ab",)),
                        LookupRequest("market")):
            with self.assertRaises(LookupUnavailable):
                OnlineGateway(fetch=fetch).lookup(request)
        self.assertEqual(fetch.sent, [])

    def test_stale_prices_are_reused_and_labelled(self):
        now = [1_800_000_000.0]
        fetch = FakeFetch()
        gateway = OnlineGateway(fetch=fetch, clock=lambda: now[0])
        request, focus = self.portfolio.request("how is Infosys doing?")
        gateway.lookup(request)
        now[0] += 3600  # the cache has expired, and now the network is down
        fetch.fail = True
        reply = self.portfolio.answer(gateway.lookup(request), focus)
        self.assertIn("I couldn't refresh them.", reply)
        self.assertIn("Infosys is at 1,023.40 rupees", reply)

    def test_assistant_answers_without_the_model(self):
        llm = FakeLLM()
        assistant, indicator, _ = make_assistant(llm=llm, portfolio=self.portfolio)
        reply, out = quietly(assistant.respond, "how is Infosys doing?")
        self.assertIn("Computed on this device: Your 10 Infosys shares", reply)
        self.assertEqual(llm.calls, [])
        self.assertIn("[ONLINE] sent only symbols=['INFY.NS', '^NSEI', '^BSESN']", out)
        self.assertIn(IndicatorState.ONLINE, indicator.states)

    def test_switched_off_market_is_answered_from_records(self):
        llm = FakeLLM("From your records, your Infosys shares were worth 18,800 rupees.")
        fetch = FakeFetch()
        gateway = OnlineGateway(fetch=fetch, kinds=("weather", "news"))
        assistant, _, _ = make_assistant(llm=llm, gateway=gateway, portfolio=self.portfolio)
        quietly(assistant.respond, "how is my monthly income doing today?")
        quietly(assistant.respond, "how is Infosys doing?")
        self.assertEqual(fetch.sent, [])

    def test_unreachable_market_falls_back_to_records_with_a_warning(self):
        assistant, _, _ = make_assistant(
            llm=FakeLLM("Your monthly income is 85000."),
            gateway=OnlineGateway(fetch=FakeFetch(fail=True)),
            portfolio=self.portfolio,
        )
        reply, _ = quietly(assistant.respond, "how is my monthly income and my investments doing today?")
        self.assertTrue(reply.startswith("I couldn't get live prices, so this is from your saved records"))


class NewsTests(unittest.TestCase):
    def test_topic_is_matched_locally_and_never_sent(self):
        fetch = FakeFetch()
        OnlineGateway(fetch=fetch).lookup(news_request("any news about TCS?").request)
        self.assertNotIn("TCS", fetch.sent_text())
        self.assertTrue(all(params == {} for _, params in fetch.sent))
        assistant, _, _ = make_assistant(gateway=OnlineGateway(fetch=FakeFetch()))
        reply, out = quietly(assistant.respond, "any news about TCS?")
        self.assertIn("the latest headlines about TCS. One: TCS builds higher bench.", reply)
        self.assertNotIn("Coal", reply)
        self.assertIn("(topic not sent)", out)

    def test_every_topic_word_must_match(self):
        assistant, _, _ = make_assistant(gateway=OnlineGateway(fetch=FakeFetch()))
        reply, _ = quietly(assistant.respond, "news about my favourite cricket team")
        self.assertTrue(reply.startswith("I found no headlines about cricket team"))

    def test_general_and_feed_specific_news(self):
        self.assertEqual(news_request("what is the news today").request.feeds, ("india", "world"))
        business = news_request("latest business news")
        self.assertEqual((business.request.feeds, business.label), (("business",), "business"))
        assistant, _, _ = make_assistant(gateway=OnlineGateway(fetch=FakeFetch()))
        reply, _ = quietly(assistant.respond, "latest business news")
        self.assertTrue(reply.startswith("From The Hindu, online: the latest business headlines. One: Coal supplies rise 36%."))

    def test_only_built_in_feeds_and_switch(self):
        with self.assertRaises(LookupUnavailable):
            OnlineGateway(fetch=FakeFetch()).lookup(LookupRequest("news", feeds=("https://example.com/rss",)))
        assistant, _, _ = make_assistant(gateway=OnlineGateway(fetch=FakeFetch(), kinds=("weather", "market")))
        reply, _ = quietly(assistant.respond, "latest news")
        self.assertIn("Online news lookups are switched off.", reply)


class CommandRecordingTests(unittest.TestCase):
    def test_command_ends_after_silence(self):
        audio = record_command(iter(frames(0, 1000, 1000, *[0] * 10, 1000)), CAPTURE, max_wait_seconds=6)
        self.assertEqual(len(audio), 1280 * 13)  # stops after 10 silent frames (0.8 s)
        self.assertEqual(audio.dtype, np.float32)

    def test_command_times_out_without_speech(self):
        audio = record_command(iter(frames(*[0] * 100)), CAPTURE, max_wait_seconds=0.4)
        self.assertEqual(audio.size, 0)

    def test_command_discarded_when_stream_stops(self):
        self.assertEqual(record_command(iter(frames(1000, 1000)), CAPTURE, max_wait_seconds=6).size, 0)


class WakePhraseTests(unittest.TestCase):
    def setUp(self):
        self.wake = WakePhrase("hey jarvis")

    def test_request_after_wake_phrase(self):
        self.assertEqual(self.wake.strip("Hey Jarvis, what's the weather today?"), "What's the weather today?")
        self.assertEqual(self.wake.strip("Jarvis, what is my EMI"), "What is my EMI")
        self.assertEqual(self.wake.strip("OK Jarvis."), "")

    def test_other_speech_is_not_a_request(self):
        for text in ("I was telling Jarvis fans", "Hey, are you coming to dinner?", "Jarvisville is nice", ""):
            self.assertIsNone(self.wake.strip(text))

    def test_custom_phrase(self):
        wake = WakePhrase("hey computer")
        self.assertEqual(wake.name, "computer")
        self.assertEqual(wake.strip("Computer, lights on"), "Lights on")

    def test_sleep_commands(self):
        self.assertTrue(is_sleep_command("That's all."))
        self.assertTrue(is_sleep_command("Stop listening!"))
        self.assertTrue(is_sleep_command("That's all, thanks."))
        self.assertTrue(is_sleep_command("OK, thank you."))
        self.assertTrue(is_sleep_command("No thanks"))
        self.assertFalse(is_sleep_command("stop the timer"))
        self.assertFalse(is_sleep_command("thanks, what is my EMI"))
        self.assertFalse(is_sleep_command(""))


class ConversationTests(unittest.TestCase):
    def test_speech_without_wake_phrase_is_ignored_and_not_printed(self):
        source, _, _, indicator = conversation(["Are you coming to dinner?"])
        result, out = quietly(source.next_utterance)
        self.assertEqual(result, "")
        self.assertFalse(source.awake)
        self.assertNotIn("dinner", out)
        self.assertEqual(indicator.states, [IndicatorState.IDLE, IndicatorState.THINKING])

    def test_wake_phrase_then_follow_ups_without_it(self):
        source, capture, _, indicator = conversation(
            ["Hey Jarvis, what is my EMI?", "And my monthly income?"]
        )
        self.assertEqual(quietly(source.next_utterance)[0], "What is my EMI?")
        self.assertTrue(source.awake)
        self.assertEqual(quietly(source.next_utterance)[0], "And my monthly income?")
        self.assertEqual(capture.waits, [None, 30])  # asleep: wait forever; awake: conversation timeout
        self.assertEqual(indicator.states[-2:], [IndicatorState.LISTENING, IndicatorState.THINKING])

    def test_bare_wake_phrase_starts_listening(self):
        source, _, _, _ = conversation(["Hey Jarvis!", "What is my EMI?"])
        self.assertEqual(quietly(source.next_utterance)[0], "")
        self.assertTrue(source.awake)
        self.assertEqual(quietly(source.next_utterance)[0], "What is my EMI?")

    def test_silence_ends_the_conversation(self):
        source, _, _, _ = conversation(["Hey Jarvis, what is my EMI?"], seconds=(1, 0))
        quietly(source.next_utterance)
        result, out = quietly(source.next_utterance)
        self.assertEqual(result, "")
        self.assertFalse(source.awake)
        self.assertIn("[SLEEP] no follow-up for 30 s", out)

    def test_sleep_command_ends_the_conversation(self):
        source, _, _, _ = conversation(["Hey Jarvis, what is my EMI?", "That's all, thanks."])
        quietly(source.next_utterance)
        source.transcriber.texts = ["That's all, thanks."]
        self.assertEqual(quietly(source.next_utterance)[0], "")
        self.assertFalse(source.awake)

    def test_wake_phrase_with_stop_never_exits_the_app(self):
        source, _, _, _ = conversation(["Hey Jarvis, stop."])
        self.assertEqual(quietly(source.next_utterance)[0], "")
        self.assertFalse(source.awake)

    def test_muting_ends_the_conversation(self):
        source, _, _, indicator = conversation(["Hey Jarvis, what is my EMI?"])
        quietly(source.next_utterance)
        source.mute.set_muted(True)
        quietly(source.next_utterance)
        self.assertFalse(source.awake)
        self.assertIs(indicator.states[-1], IndicatorState.MUTED)

    def test_only_the_start_of_long_speech_is_checked_while_asleep(self):
        source, _, transcriber, _ = conversation(["We should book the tickets soon"], seconds=(10,))
        quietly(source.next_utterance)
        self.assertEqual(transcriber.lengths, [3.0])  # 3 s checked, the rest never transcribed

    def test_long_request_is_transcribed_in_full_after_waking(self):
        source, _, transcriber, _ = conversation(
            ["Hey Jarvis, when does", "Hey Jarvis, when does my car insurance expire?"], seconds=(6,)
        )
        self.assertEqual(quietly(source.next_utterance)[0], "When does my car insurance expire?")
        self.assertEqual(transcriber.lengths, [3.0, 6.0])

    def test_without_wake_phrase_every_utterance_is_a_request(self):
        indicator = RecordingIndicator()
        source = MicInput(FakeCapture(), FakeTranscriber(), SoftwareMuteSwitch(), indicator)
        self.assertEqual(quietly(source.next_utterance)[0], "what is my income")
        self.assertEqual(indicator.states, [IndicatorState.LISTENING, IndicatorState.THINKING])


class MuteAndIndicatorTests(unittest.TestCase):
    def test_muted_mic_never_records(self):
        capture = FakeCapture()
        indicator = RecordingIndicator()
        source = MicInput(capture, FakeTranscriber(), SoftwareMuteSwitch(initial_muted=True), indicator)
        self.assertEqual(source.next_utterance(), "")
        self.assertEqual(capture.calls, 0)
        self.assertEqual(indicator.states, [IndicatorState.MUTED])

    def test_mute_switch_toggles(self):
        switch = SoftwareMuteSwitch()
        self.assertFalse(switch.is_muted())
        self.assertTrue(switch.toggle())
        self.assertTrue(switch.is_muted())

    def test_console_indicator_only_prints_changes(self):
        lines = []
        indicator = ConsoleIndicator(output=lines.append)
        indicator.show(IndicatorState.IDLE)
        indicator.show(IndicatorState.IDLE)
        indicator.show(IndicatorState.LISTENING)
        self.assertEqual(lines, ["[LED] IDLE", "[LED] LISTENING"])


class SpeechOutputTests(unittest.TestCase):
    def test_espeak_renders_wav_then_plays_through_alsa_device(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs["input"]))
            return subprocess.CompletedProcess(command, 0, stdout=b"RIFF-wav", stderr=b"")

        speaker = EspeakSpeaker(device="plughw:CARD=Headphones", run=fake_run)
        with contextlib.redirect_stdout(io.StringIO()):
            speaker.say("-v is not an option here")
        (synth, synth_input), (play, play_input) = calls
        self.assertEqual(synth[:3], ["espeak-ng", "--stdin", "--stdout"])
        self.assertEqual(synth_input, b"-v is not an option here")
        self.assertEqual(play, ["aplay", "-q", "-D", "plughw:CARD=Headphones"])
        self.assertEqual(play_input, b"RIFF-wav")

    def test_playback_failure_is_reported_not_raised(self):
        def failing_run(command, **kwargs):
            raise subprocess.CalledProcessError(1, command, stderr=b"audio open error: Unknown error 524")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            EspeakSpeaker(run=failing_run).say("hello")
        self.assertIn("Unknown error 524", out.getvalue())
        self.assertIn("AUDIO_OUTPUT_DEVICE", out.getvalue())


if __name__ == "__main__":
    unittest.main()
