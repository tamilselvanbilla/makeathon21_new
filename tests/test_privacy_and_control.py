import ast
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from companion.brain import knowledge, policy  # noqa: E402
from companion.brain.llm import clean_model_response  # noqa: E402
from companion.brain.router import Intent, route  # noqa: E402
from companion.device.indicator import ConsoleIndicator, IndicatorState  # noqa: E402
from companion.device.mute import SoftwareMuteSwitch  # noqa: E402
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
        {
            "owner": "John",
            "record_type": "income",
            "source": "Test sample",
            "monthly_income": 85000,
        }
    ],
    "medical": [],
    "documents": [],
    "history": [],
}


def make_assistant(llm=None, gateway=None):
    indicator = RecordingIndicator()
    speaker = FakeSpeaker()
    assistant = Assistant(
        llm=llm or FakeLLM(),
        records=SAMPLE_RECORDS,
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

    def test_knowledge_matches_are_readable_prose(self):
        result = knowledge.find_relevant_records("income for John", SAMPLE_RECORDS)
        self.assertIn("monthly income", result.casefold())
        self.assertIn("85000", result)
        self.assertNotIn("{", result)
        self.assertNotIn("[history]", result)


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


if __name__ == "__main__":
    unittest.main()
