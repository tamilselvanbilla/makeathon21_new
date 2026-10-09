import ast
import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from companion.brain import policy  # noqa: E402
from companion.brain.knowledge import KnowledgeBase  # noqa: E402
from companion.brain.prompts import NOT_IN_RECORDS_ANSWER  # noqa: E402
from companion.brain.llm import clean_model_response  # noqa: E402
from companion.brain.router import Intent, route  # noqa: E402
from companion.device.indicator import ConsoleIndicator, IndicatorState  # noqa: E402
from companion.device.mute import SoftwareMuteSwitch  # noqa: E402
from companion.device.tts import EspeakSpeaker  # noqa: E402
from companion.online_gateway import LookupUnavailable, OnlineGateway  # noqa: E402
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
    def __init__(self):
        self.calls = 0

    def record_utterance(self, should_abort):
        self.calls += 1
        return np.ones(1600, dtype=np.float32)


class FakeTranscriber:
    def transcribe(self, audio):
        return "what is my income"


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


def make_assistant(llm=None, gateway=None):
    indicator = RecordingIndicator()
    speaker = FakeSpeaker()
    assistant = Assistant(
        llm=llm or FakeLLM(),
        knowledge=KnowledgeBase(SAMPLE_RECORDS),
        gateway=gateway or OnlineGateway(enabled=True),
        indicator=indicator,
        speaker=speaker,
    )
    return assistant, indicator, speaker


class PrivacyTests(unittest.TestCase):
    def test_lookup_classification(self):
        self.assertTrue(policy.is_allowed_cloud_lookup("weather in bengaluru"))
        self.assertFalse(policy.is_allowed_cloud_lookup("explain my financial data"))

    def test_gateway_only_accepts_text(self):
        with self.assertRaises(TypeError):
            OnlineGateway().lookup(b"\x00\x01raw-audio")
        with self.assertRaises(TypeError):
            OnlineGateway().lookup(np.zeros(16000, dtype=np.float32))

    def test_gateway_refuses_private_or_disabled_requests(self):
        with self.assertRaises(LookupUnavailable):
            OnlineGateway().lookup("search my medical records")
        with self.assertRaises(LookupUnavailable):
            OnlineGateway(enabled=False).lookup("weather in bengaluru")

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
        assistant, indicator, _ = make_assistant(llm=llm)
        reply = assistant.respond("what is the weather in bengaluru")
        self.assertIn("won't guess", reply)
        self.assertEqual(llm.calls, [])
        self.assertIn(IndicatorState.ONLINE, indicator.states)

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


class MuteAndIndicatorTests(unittest.TestCase):
    def test_muted_mic_never_records(self):
        capture = FakeCapture()
        indicator = RecordingIndicator()
        source = MicInput(capture, FakeTranscriber(), SoftwareMuteSwitch(initial_muted=True), indicator)
        self.assertEqual(source.next_utterance(), "")
        self.assertEqual(capture.calls, 0)
        self.assertEqual(indicator.states, [IndicatorState.MUTED])

    def test_unmuted_mic_shows_listening_then_thinking(self):
        indicator = RecordingIndicator()
        source = MicInput(FakeCapture(), FakeTranscriber(), SoftwareMuteSwitch(), indicator)
        self.assertEqual(source.next_utterance(), "what is my income")
        self.assertEqual(indicator.states, [IndicatorState.LISTENING, IndicatorState.THINKING])

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
