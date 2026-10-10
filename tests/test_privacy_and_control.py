import ast
import json
import contextlib
import io
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from companion.brain import policy  # noqa: E402
from companion.brain.knowledge import KnowledgeBase, load_knowledge  # noqa: E402
from companion.brain.memory import (  # noqa: E402
    ConversationMemory,
    MemoryCommand,
    note_answer,
    parse_memory_command,
)
from companion.brain.prompts import NO_NOTE_ANSWER, NO_RECORDS, NOT_IN_RECORDS_ANSWER, build_system_prompt, build_welcome  # noqa: E402
from companion.brain.llm import clean_model_response  # noqa: E402
from companion.audio.capture import pick_microphone, record_command  # noqa: E402
from companion.audio.stt import echoes_prompt, looks_like_noise, reliable  # noqa: E402
from companion.brain.router import Intent, extract_place, parse_lookup, route  # noqa: E402
from companion.config import CaptureConfig  # noqa: E402
from companion.device.indicator import ConsoleIndicator, IndicatorState  # noqa: E402
from companion.device.mute import SoftwareMuteSwitch  # noqa: E402
from companion.device.tts import EspeakSpeaker, TTS_RATE  # noqa: E402
from companion.brain.market import Portfolio  # noqa: E402
from companion.brain.news import news_request  # noqa: E402
from companion.online_gateway import (  # noqa: E402
    FORECAST_URL,
    GEOCODING_URL,
    best_place,
    LookupRequest,
    LookupUnavailable,
    OnlineGateway,
    check_host_allowed,
)
from companion.pipeline import Assistant, MicInput  # noqa: E402
from companion.telemetry import timed_event  # noqa: E402
from companion.tracing import Tracer, annotate, span  # noqa: E402

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


def make_assistant(llm=None, gateway=None, memory=None, portfolio=None, tracer=None):
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
        tracer=tracer,
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

    def test_best_place_prefers_home_country_then_population(self):
        results = [
            {"name": "Bangalore Town", "country_code": "PK"},
            {"name": "Mysore Road Tolgate", "country_code": "IN"},
            {"name": "Mysuru", "country_code": "IN", "population": 920550},
        ]
        self.assertEqual(best_place(results, "IN")["name"], "Mysuru")
        self.assertEqual(best_place(results[:1], "IN")["name"], "Bangalore Town")  # nothing better available

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
        self.assertTrue(policy.is_uncertain("I cannot answer this accurately"))
        self.assertTrue(policy.is_uncertain("   "))
        self.assertFalse(policy.is_uncertain("Here is the answer"))
        self.assertEqual(policy.require_local_answer("  "), policy.LOCAL_FALLBACK_ANSWER)
        self.assertEqual(policy.require_local_answer("It is not recorded."), "It is not recorded.")

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
        self.assertEqual(extract_place("weather in Mysore and should I go for a run?"), "Mysuru")  # old name mapped
        self.assertEqual(extract_place("What is a weather in Bangalore today?"), "Bengaluru")
        self.assertIsNone(extract_place("should I go for a run"))

    def test_rain_matches_whole_words_only(self):
        self.assertIs(route("will it rain in Chennai"), Intent.ONLINE_LOOKUP)
        self.assertIs(route("when is my train"), Intent.LOCAL_REASONING)
        self.assertIs(route("where is my umbrella?"), Intent.LOCAL_REASONING)
        self.assertIs(route("do I need an umbrella tomorrow?"), Intent.ONLINE_LOOKUP)

    def test_replies_speak_to_the_user(self):
        self.assertEqual(clean_model_response("The validity period of my insurance ends in 2027."),
                         "The validity period of your insurance ends in 2027.")
        self.assertEqual(clean_model_response("My EMI is INR 9200."), "Your EMI is INR 9200.")
        self.assertEqual(clean_model_response("Let me know if you need more."), "Let me know if you need more.")

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
        self.assertEqual(len(KnowledgeBase(SAMPLE_RECORDS).search("do I have insurance")), 2)
        self.assertEqual(len(KnowledgeBase(SAMPLE_RECORDS, top_k=1).search("do I have insurance")), 1)

    def test_fts_operators_in_questions_are_harmless(self):
        self.assertEqual(self.kb.search('income" OR NOT * ('), self.kb.search("income"))

    def test_personal_question_detection(self):
        self.assertTrue(self.kb.is_personal("where did I park"))
        self.assertFalse(self.kb.is_personal("what is the capital of France"))


class RetrievalBenchmarkTests(unittest.TestCase):
    """Natural spoken questions over the real records (tests/data/knowledge_retrieval_eval.json)."""

    def test_every_question_finds_the_right_record_or_nothing(self):
        kb = KnowledgeBase(load_knowledge())
        failures = []
        for case in json.loads((ROOT / "tests" / "data" / "knowledge_retrieval_eval.json").read_text())["cases"]:
            if case.get("known_miss"):
                continue
            hits = kb.search(case["q"])
            top = hits[0].text.split(";")[0].replace("Record Type: ", "") if hits else None
            if (top is not None) if case["expect"] is None else (top not in case["expect"]):
                failures.append(f"{case['q']} -> {top} (expected {case['expect']})")
        self.assertEqual(failures, [])


class MicrophoneSelectionTests(unittest.TestCase):
    MAC = [(0, "MacBook Pro Microphone")]
    PI = [(0, "bcm2835 Headphones: - (hw:0,0)"), (1, "Monitor of Built-in Audio"), (2, "USB PnP Sound Device: Audio (hw:2,0)")]

    def test_system_default_is_used_without_asking(self):
        self.assertEqual(pick_microphone(self.MAC, 0, None), 0)

    def test_without_a_default_a_likely_microphone_is_chosen(self):
        self.assertEqual(pick_microphone(self.PI, None, None), 2)  # USB mic, not the monitor or headphone jack

    def test_mic_device_by_index_or_name(self):
        self.assertEqual(pick_microphone(self.PI, 0, "2"), 2)
        self.assertEqual(pick_microphone(self.PI, 0, "usb"), 2)
        with self.assertRaises(RuntimeError):
            pick_microphone(self.PI, 0, "7")
        with self.assertRaises(RuntimeError):
            pick_microphone(self.PI, 0, "ReSpeaker")


class SpeechFilterTests(unittest.TestCase):
    def test_repetitive_noise_transcripts_are_dropped(self):
        self.assertTrue(looks_like_noise("M.D. M.D. M.D. You have to back up. M.D. S.B. S.B. S.B. S.B. S.B. S.B."))
        self.assertFalse(looks_like_noise("Can you give me what are the medications I am taking daily?"))
        self.assertFalse(looks_like_noise("no no no no"))  # short answers are kept

    def test_hint_word_echo_is_dropped(self):
        self.assertTrue(echoes_prompt("EMI PAN Aadhaar", "EMI PAN Aadhaar Bengaluru"))
        self.assertFalse(echoes_prompt("What is my EMI?", "EMI PAN Aadhaar Bengaluru"))
        self.assertFalse(echoes_prompt("What is my PAN card number?", "EMI PAN Aadhaar Bengaluru"))

    def test_whisper_quality_signals(self):
        segment = lambda **kw: SimpleNamespace(**{"no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.2, **kw})  # noqa: E731
        self.assertTrue(reliable(segment()))
        self.assertFalse(reliable(segment(no_speech_prob=0.8, avg_logprob=-1.4)))  # probably silence
        self.assertFalse(reliable(segment(compression_ratio=3.1)))  # degenerate repetition


class PromptTests(unittest.TestCase):
    def test_system_prompt_describes_role_and_records(self):
        kb = KnowledgeBase(load_knowledge())
        prompt = build_system_prompt(kb.primary_user, kb.currency, "Sam", kb.family)
        for part in ("personal assistant for John", "insurance policies", "medical and health records",
                     "identity documents", "use only Knowledge and Memory",
                     "general questions", "identity numbers only when asked"):
            self.assertIn(part, prompt)
        self.assertIn("Prior assistant replies in Memory are conversation history, not verified facts", prompt)
        self.assertNotIn("wife Jane", prompt)
        self.assertNotIn("son Robert", prompt)

    def test_welcome_names_the_assistant(self):
        self.assertEqual(build_welcome("Sam", "John"), "Hello John, I'm Sam, your private assistant. Ask me anything.")


class PipelineTests(unittest.TestCase):
    def test_local_question_uses_local_model_and_knowledge(self):
        llm = FakeLLM()
        assistant, indicator, _ = make_assistant(llm=llm)
        reply = assistant.respond("what is my monthly income")
        self.assertEqual(reply, "Your monthly income is 85000 rupees.")
        self.assertIn("85000", llm.calls[0])
        self.assertIn(IndicatorState.THINKING, indicator.states)
        self.assertNotIn(IndicatorState.ONLINE, indicator.states)

    def test_long_replies_are_cut_to_fifty_words_at_a_sentence(self):
        long = "Your EMI is INR 9,200 a month. " + "You also hold Infosys shares and a mutual fund. " * 10
        assistant, _, _ = make_assistant(llm=FakeLLM(long))
        reply = assistant.respond("what is my emi amount")
        self.assertLessEqual(len(reply.split()), 50)
        self.assertTrue(reply.startswith("Your EMI is INR 9,200 a month.") and reply.endswith("."))

    def test_word_limit_does_not_split_decimals_or_keep_unfinished_sentences(self):
        cut = policy.limit_words("Your BMI is 25.5 and " + "very " * 60, 50)
        self.assertEqual(len(cut.split()), 50)
        self.assertTrue(cut.startswith("Your BMI is 25.5 and very") and cut.endswith("very."))
        self.assertEqual(policy.limit_words("Your BMI is 25.5. " + "Walk more " * 30, 50), "Your BMI is 25.5.")
        self.assertEqual(policy.full_sentences("Your BMI is 25.5. You should eat better and"), "Your BMI is 25.5.")
        self.assertEqual(policy.full_sentences("Your EMI is INR 9,200"), "Your EMI is INR 9,200.")
        self.assertEqual(policy.full_sentences('You said "hi."'), 'You said "hi."')

    def test_example_figure_is_allowed_when_a_record_holds_it(self):
        llm = FakeLLM("Your bond has a face value of INR 50,000.")
        assistant, _, _ = make_assistant(llm=llm)
        assistant.knowledge = KnowledgeBase({"financial": [{"owner": "John", "record_type": "bond", "face_value": 50000}]})
        self.assertEqual(assistant.respond("what is my bond face value"), "Your bond has a face value of INR 50,000.")

    def test_emi_question_sends_only_the_loan_record(self):
        llm = FakeLLM()
        assistant, _, _ = make_assistant(llm=llm)
        assistant.respond("What is my E.M.I. amount?")
        self.assertIn("9200", llm.calls[0])
        self.assertNotIn("INFY", llm.calls[0])

    def test_medicine_answer_spoken_as_the_user_is_turned_to_you(self):
        self.assertEqual(policy.as_second_person("I am taking Sample medication."), "You are taking Sample medication.")
        self.assertEqual(policy.as_second_person("I take it daily."), "You take it daily.")
        self.assertEqual(policy.as_second_person("I am 175 cm tall."), "You are 175 cm tall.")
        self.assertEqual(policy.as_second_person("I cannot find that."), "I cannot find that.")
        self.assertEqual(policy.as_second_person("I am not sure."), "I am not sure.")
        self.assertEqual(policy.as_second_person("I'm sorry, that isn't recorded."), "I'm sorry, that isn't recorded.")

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
        self.assertEqual(NOT_IN_RECORDS_ANSWER, "No matched data found.")
        self.assertEqual(assistant.respond("what is my blood group"), NOT_IN_RECORDS_ANSWER)
        self.assertEqual(assistant.respond("when does my passport expire"), NOT_IN_RECORDS_ANSWER)
        self.assertEqual(llm.calls, [])

    def test_removed_family_data_is_not_answered_from_the_model(self):
        llm = FakeLLM()
        assistant, _, _ = make_assistant(llm=llm)
        assistant.knowledge = KnowledgeBase(load_knowledge())
        self.assertEqual(assistant.respond("what is my wife's employer"), NOT_IN_RECORDS_ANSWER)
        self.assertEqual(llm.calls, [])

    def test_general_question_without_records_still_reaches_the_model(self):
        llm = FakeLLM("Paris.")
        assistant, _, _ = make_assistant(llm=llm)
        self.assertEqual(assistant.respond("what is the capital of France"), "Paris.")
        self.assertIn(f"Knowledge: {NO_RECORDS}", llm.calls[0])

    def test_context_debug_prints_what_the_model_receives(self):
        assistant, _, _ = make_assistant()
        assistant.debug_context = True
        _, out = quietly(assistant.respond, "what is my monthly income")
        self.assertIn("[CONTEXT] records sent to the model:", out)
        self.assertIn("Monthly Income: 85000", out)

    def test_copied_prompt_example_is_not_presented_as_data(self):
        assistant, _, _ = make_assistant(llm=FakeLLM("A. Your monthly salary is INR 50,000."))
        reply, _ = quietly(assistant.respond, "A")
        self.assertEqual(reply, "Sorry, I didn't catch that. Could you say it again?")

    def test_uncertain_model_answer_is_spoken_but_not_remembered(self):
        memory = ConversationMemory(clock=Clock())
        assistant, _, _ = make_assistant(llm=FakeLLM("I do not know."), memory=memory)
        self.assertEqual(assistant.respond("what is my income"), "I do not know.")
        self.assertIsNone(memory.last_turn())

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
            self.memory = ConversationMemory(path, clock=self.clock)
            self.memory.add_turn("what is my EMI", "Your EMI is INR 9200.")
            self.clock.advance(days=31)
            restarted = ConversationMemory(path, clock=self.clock)
            # Notes stay until the user deletes them; past turns expire.
            self.assertIn("B2", restarted.context_for("where did I park"))
            self.assertEqual(restarted.context_for("what did you tell me about my EMI?"), "")


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


class NoteAndScheduleTests(unittest.TestCase):
    """The cases from a Raspberry Pi session where notes were lost or not recognised."""

    def setUp(self):
        self.clock = Clock(datetime(2026, 10, 9, 21, 15))  # Friday evening
        self.memory = ConversationMemory(clock=self.clock)
        self.llm = FakeLLM()
        self.assistant, _, self.speaker = make_assistant(llm=self.llm, memory=self.memory)

    def say(self, text):
        return quietly(self.assistant.respond, text)[0]

    def test_where_did_i_park_finds_a_note_without_the_word_park(self):
        self.say("Remember that I have got my car on level B2.")
        self.assertEqual(self.say("Where did I park my car?"), "On 9 October you told me that you have got your car on level B2.")

    def test_note_outranks_a_past_turn_repeating_it(self):
        self.say("Remember that I have got my car on level B2.")
        self.say("Where did I park my car?")  # stored as a turn
        self.assertEqual(self.say("Where did I park my car?"), "On 9 October you told me that you have got your car on level B2.")

    def test_forget_that_removes_the_note_just_discussed_and_its_quotes(self):
        self.say("Remember that my gym locker code is 4512")
        self.say("What is my gym locker code?")  # answer quoting the note is stored as a turn
        self.assertEqual(self.say("Forget that"), "Okay, I've forgotten that your gym locker code is 4512.")
        self.assertNotIn("4512", self.say("What is my gym locker code?"))
        self.assertNotIn("4512", self.memory.context_for("what did you tell me about my gym locker code?"))

    def test_where_is_my_thing_without_a_note_is_not_guessed(self):
        self.assertEqual(self.say("Where is the spare key?"), NO_NOTE_ANSWER)
        self.assertEqual(self.say("Where did I keep my glasses?"), NO_NOTE_ANSWER)
        self.assertEqual(self.llm.calls, [])
        self.say("Where is the Eiffel Tower?")  # a name: general knowledge, the model answers
        self.assertEqual(len(self.llm.calls), 1)

    def test_family_words_match_notes(self):
        self.say("Remember that mom's birthday is on 12 March")
        self.assertEqual(self.say("When is my mother's birthday?"), "On 9 October you told me that mom's birthday is on 12 March.")

    def test_tasks_are_offered(self):
        self.assertEqual(self.say("My wife asked me to buy vegetables on the way home"),
                         "Do you want me to remember that your wife asked you to buy vegetables on the way home?")

    def test_bare_remember_waits_for_the_content(self):
        self.assertEqual(self.say("Please remember that."), "Sure. What should I remember?")
        reply = self.say("meeting with Jay tomorrow at 7am.")
        self.assertEqual(reply, "Okay, I'll remember: meeting with Jay on Saturday 10 October at 7:00 AM, and remind you then.")
        self.assertEqual(self.say("When is my meeting with Jay?"),
                         "On 9 October you told me: meeting with Jay on Saturday 10 October at 7:00 AM.")
        self.assertEqual(self.say("When is the meeting?"), "Tomorrow, 10 October: meeting with Jay at 7:00 AM.")
        self.assertEqual(self.llm.calls, [])

    def test_remind_me_to(self):
        reply = self.say("Remind me to call mom at 6 pm tomorrow")
        self.assertEqual(reply, "Okay, I'll remember that you need to call mom on Saturday 10 October at 6:00 PM, and remind you then.")
        self.assertEqual(self.say("Remind me"), "Sure. What should I remind you to do?")
        self.assertIn("you need to buy milk", self.say("buy milk"))

    def test_statement_with_a_date_is_offered_and_saved_on_yes(self):
        self.assertEqual(self.say("I have a meeting with Jay tomorrow."), "Do you want me to remember that you have a meeting with Jay tomorrow?")
        self.assertEqual(self.say("Yes please"), "Okay, I'll remember that you have a meeting with Jay on Saturday 10 October.")
        self.assertEqual(self.llm.calls, [])

    def test_statement_declined_or_ignored_is_not_saved(self):
        self.say("I parked on level B2")
        self.assertEqual(self.say("No"), "Okay, I won't remember it.")
        self.say("I lent my drill to Ravi")
        self.say("what is my EMI")  # moved on: answered normally
        self.assertEqual(self.memory.search("who has my drill"), [])
        self.assertEqual(len(self.llm.calls), 1)

    def test_questions_are_not_saved_as_notes(self):
        self.assertIsNone(parse_memory_command("Do you remember where I parked?"))
        self.assertIsNone(parse_memory_command("remember where I parked?"))

    def test_schedule_for_a_day(self):
        self.say("Remember I have a meeting with Jay tomorrow at 7am")
        self.say("Remind me to pay rent on Monday")
        self.assertEqual(self.say("What is my schedule looks like tomorrow?"),
                         "Tomorrow, 10 October: you have a meeting with Jay at 7:00 AM.")
        self.assertEqual(self.say("Do I have any meetings today?"), "You have nothing noted for today, 9 October.")
        self.assertEqual(self.say("What are my reminders?"),
                         "Tomorrow, 10 October: you have a meeting with Jay at 7:00 AM. "
                         "Monday, 12 October: you need to pay rent.")
        # A question with a subject is answered from the note, not as a schedule.
        self.assertIn("7:00 AM", self.say("What time is my meeting with Jay?"))

    def test_due_reminder_is_announced_once(self):
        self.say("Remind me to take my tablet in 10 minutes")
        self.assertEqual(self.assistant.due_announcement(), "")
        self.clock.advance(minutes=10)
        self.assertEqual(self.assistant.due_announcement(), "Reminder: you need to take your tablet at 9:25 PM.")
        self.assertEqual(self.assistant.due_announcement(), "")

    def test_reminders_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.sqlite3"
            ConversationMemory(path, clock=self.clock).add_note("I have a meeting with Jay tomorrow at 7am")
            self.clock.advance(hours=9)  # 6:15 next morning, device restarted
            assistant, _, _ = make_assistant(memory=ConversationMemory(path, clock=self.clock))
            self.assertEqual(assistant.briefing(), "Today you have: you have a meeting with Jay at 7:00 AM.")
            self.clock.advance(minutes=45)
            self.assertEqual(assistant.due_announcement(), "Reminder: you have a meeting with Jay at 7:00 AM.")
            self.assertEqual(ConversationMemory(path, clock=self.clock).due_reminders(), [])

    def test_model_cannot_claim_to_remember(self):
        self.llm.reply = "Okay, I will keep this in mind."
        reply = self.say("The plumber is coming on Monday")
        self.assertEqual(reply, "I haven't saved that. Do you want me to remember that the plumber is coming on Monday?")
        self.assertEqual(self.say("yes"), "Okay, I'll remember that the plumber is coming on Monday 12 October.")

    def test_forget_removes_the_reminder(self):
        self.say("Remind me to call mom in 5 minutes")
        self.say("forget that")
        self.clock.advance(minutes=5)
        self.assertEqual(self.assistant.due_announcement(), "")


class TracingTests(unittest.TestCase):
    def run_session(self, *lines, tracer=None, source=None, llm=None):
        tracer = tracer or Tracer()
        assistant, _, _ = make_assistant(llm=llm, memory=ConversationMemory(clock=Clock()), tracer=tracer)
        quietly(assistant.run, source or ScriptedInput(*lines))
        return tracer

    def test_spans_outside_a_turn_do_nothing(self):
        with span("stt") as current:
            annotate(x=1)
        self.assertIsNone(current)

    def test_nested_spans_and_timed_events(self):
        tracer = Tracer()
        with quietly_turn(tracer):
            with span("outer"):
                with timed_event("llm_generation", model="m"):
                    annotate(completion_tokens=5)
        trace = tracer.get(tracer.recent(1)[0]["id"])
        outer, inner = trace["spans"]
        self.assertEqual((outer["name"], inner["name"], inner["parent"]), ("outer", "llm_generation", 0))
        self.assertEqual(inner["attrs"], {"model": "m", "completion_tokens": 5})

    def test_each_turn_is_traced_with_route_and_steps(self):
        tracer = self.run_session("what is my monthly income", "Remember that I parked on level B2", "what is my blood group")
        turns = list(reversed(tracer.recent()))
        self.assertEqual([t["route"] for t in turns], ["llm", "memory_remember", "not_in_records"])
        self.assertEqual(turns[0]["input"], "what is my monthly income")
        self.assertEqual(turns[0]["reply"], "Your monthly income is 85000 rupees.")
        names = [s["name"] for s in tracer.get(turns[0]["id"])["spans"]]
        self.assertEqual(names, ["knowledge", "memory", "speak"])
        self.assertIsNotNone(turns[0]["response_ms"])

    def test_traces_name_records_but_never_contain_them(self):
        tracer = self.run_session("what is my monthly income")
        trace = tracer.get(tracer.recent(1)[0]["id"])
        knowledge = trace["spans"][0]["attrs"]
        self.assertGreaterEqual(knowledge["records"], 1)
        self.assertIn("financial/John", knowledge["sources"])
        self.assertNotIn("85000", json.dumps(trace["spans"]))

    def test_question_text_can_be_left_out(self):
        tracer = self.run_session("what is my monthly income", tracer=Tracer(content=False))
        turn = tracer.recent(1)[0]
        self.assertEqual((turn["input"], turn["reply"], turn["route"]), ("", "", "llm"))

    def test_forget_everything_deletes_traces(self):
        tracer = self.run_session("what is my monthly income", "forget everything")
        self.assertEqual([t["route"] for t in tracer.recent()], ["memory_forget_all"])

    def test_silent_timeouts_are_dropped_and_ignored_speech_kept(self):
        class Limited:  # MicInput never ends by itself
            def __init__(self, source, turns):
                self.source, self.turns = source, turns

            def next_utterance(self):
                self.turns -= 1
                return self.source.next_utterance() if self.turns >= 0 else None

        mic = MicInput(FakeCapture(0, 1.0, 1.0), FakeTranscriber("A", "what is my monthly income"),
                       SoftwareMuteSwitch(), RecordingIndicator())
        tracer = self.run_session(source=Limited(mic, 3))
        turns = tracer.recent()
        self.assertEqual([t["route"] for t in turns], ["llm", "ignored"])  # the silent wait is dropped
        self.assertEqual(turns[1]["attrs"]["ignored"], "too short")
        self.assertEqual(turns[1]["input"], "")  # ignored speech is never stored
        spans = [s["name"] for s in tracer.get(turns[0]["id"])["spans"]]
        self.assertEqual(spans[0], "listen")
        self.assertGreater(turns[0]["response_ms"], 0)

    def test_turn_failure_is_recorded(self):
        class Broken:
            def chat(self, system, user):
                raise RuntimeError("boom")
        tracer = self.run_session("what is my monthly income", llm=Broken())
        self.assertEqual(tracer.recent(1)[0]["status"], "error")

    def test_persists_and_reports_stats(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "traces.sqlite3"
            self.run_session("what is my monthly income", "what is my blood group", tracer=Tracer(path))
            stats = Tracer(path).stats()
            self.assertEqual(stats["turns"], 2)
            self.assertEqual(stats["routes"], {"llm": 1, "not_in_records": 1})
            self.assertEqual(stats["steps"]["speak"]["n"], 2)
            out = subprocess.run([sys.executable, str(ROOT / "scripts" / "traces.py"), "--file", str(path), "show", "last"],
                                 capture_output=True, text=True, check=True).stdout
            self.assertIn("route=not_in_records", out)
            self.assertIn("knowledge", out)


def quietly_turn(tracer):
    return tracer.turn()


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

    def test_follow_up_on_a_new_topic_is_not_sent_down_the_market_route(self):
        llm = FakeLLM("Your EMI is INR 9,200 a month.")
        memory = ConversationMemory(clock=Clock())
        assistant, _, _ = make_assistant(llm=llm, memory=memory, portfolio=self.portfolio)
        quietly(assistant.respond, "how are my investments doing today?")
        reply, _ = quietly(assistant.respond, "What about my EMI amount?")
        self.assertEqual(reply, "Your EMI is INR 9,200 a month.")
        self.assertIn("9200", llm.calls[-1])
        self.assertNotIn("Previous question", llm.calls[-1])

    def test_follow_up_naming_a_holding_stays_on_the_market_route(self):
        llm = FakeLLM()
        memory = ConversationMemory(clock=Clock())
        assistant, _, _ = make_assistant(llm=llm, memory=memory, portfolio=self.portfolio)
        quietly(assistant.respond, "how is the Nifty doing?")
        reply, _ = quietly(assistant.respond, "and Infosys?")
        self.assertIn("Infosys", reply)
        self.assertEqual(llm.calls, [])

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
        audio = record_command(iter(frames(0, 1000, 1000, 1000, *[0] * 10, 1000)), CAPTURE, max_wait_seconds=6)
        self.assertEqual(len(audio), 1280 * 14)  # pre-roll + 3 loud + 10 silent frames (0.8 s)
        self.assertEqual(audio.dtype, np.float32)

    def test_brief_noise_spikes_do_not_start_a_recording(self):
        spikes = frames(1000, 0, 0, 1000, 1000, 0, *[0] * 20)  # 1- and 2-frame bursts
        self.assertEqual(record_command(iter(spikes), CAPTURE, max_wait_seconds=1.5).size, 0)

    def test_only_half_a_second_before_speech_is_kept(self):
        audio = record_command(iter(frames(*[100] * 60, 1000, 1000, 1000, *[0] * 10)), CAPTURE, max_wait_seconds=30)
        # 4.8 s of room noise before speech: only the 6 frames of pre-roll + 3 onset frames survive
        self.assertEqual(len(audio), 1280 * (9 + 10))

    def test_max_record_counts_from_speech_start(self):
        audio = record_command(iter(frames(*[0] * 100, *[1000] * 400)), CAPTURE, max_wait_seconds=30)
        self.assertLessEqual(len(audio) / 16000, 15 + 0.8)  # 15 s of speech plus pre-roll

    def test_command_times_out_without_speech(self):
        audio = record_command(iter(frames(*[0] * 100)), CAPTURE, max_wait_seconds=0.4)
        self.assertEqual(audio.size, 0)

    def test_command_discarded_when_stream_stops(self):
        self.assertEqual(record_command(iter(frames(1000, 1000)), CAPTURE, max_wait_seconds=6).size, 0)


class MuteAndIndicatorTests(unittest.TestCase):
    def test_one_word_noise_transcripts_are_ignored(self):
        indicator = RecordingIndicator()
        source = MicInput(FakeCapture(), FakeTranscriber("A"), SoftwareMuteSwitch(), indicator)
        self.assertEqual(quietly(source.next_utterance)[0], "")

    def test_every_utterance_is_a_request(self):
        indicator = RecordingIndicator()
        source = MicInput(FakeCapture(), FakeTranscriber(), SoftwareMuteSwitch(), indicator)
        self.assertEqual(quietly(source.next_utterance)[0], "what is my income")
        self.assertEqual(indicator.states, [IndicatorState.LISTENING, IndicatorState.THINKING])


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
        self.assertEqual(synth[synth.index("-s") + 1], str(TTS_RATE))
        self.assertEqual(synth_input, b"-v is not an option here")
        self.assertEqual(play, ["aplay", "-q", "-D", "plughw:CARD=Headphones"])
        self.assertEqual(play_input, b"RIFF-wav")

    def test_playback_failure_is_reported_not_raised(self):
        def run(command, **kwargs):
            if command[0] == "aplay":
                raise subprocess.CalledProcessError(1, command, stderr=b"audio open error: Unknown error 524")
            return subprocess.CompletedProcess(command, 0, stdout=b"RIFF-wav", stderr=b"")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            EspeakSpeaker(["default"], run=run).say("hello")
        self.assertIn("Unknown error 524", out.getvalue())
        self.assertIn("AUDIO_OUTPUT_DEVICE", out.getvalue())

    def test_falls_back_to_a_working_device_and_keeps_it(self):
        played = []

        def run(command, **kwargs):
            if command[0] == "aplay":
                played.append(command[-1])
                if command[-1] == "default":
                    raise subprocess.CalledProcessError(1, command, stderr=b"Unknown error 524")
            return subprocess.CompletedProcess(command, 0, stdout=b"RIFF-wav", stderr=b"")

        speaker = EspeakSpeaker(["default", "plughw:3,0"], run=run)
        with contextlib.redirect_stdout(io.StringIO()):
            speaker.say("one")
            speaker.say("two")
        self.assertEqual(played, ["default", "plughw:3,0", "plughw:3,0"])  # second reply goes straight there
        self.assertEqual(speaker.device, "plughw:3,0")

    def test_output_prefers_the_microphones_own_device(self):
        from companion.device.tts import output_candidates
        self.assertEqual(
            output_candidates("Plantronics Blackwire 3220 Seri: USB Audio (hw:3,0)"),
            ["plughw:3,0", "plughw:CARD=Headphones,DEV=0", "default"],
        )


if __name__ == "__main__":
    unittest.main()
