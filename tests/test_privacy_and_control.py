import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


class RequirementTests(unittest.TestCase):
    def test_mute_controls_are_exposed(self):
        control = importlib.import_module("control")
        self.assertFalse(control.is_muted(False))
        self.assertTrue(control.is_muted(True))

    def test_audio_never_leaves_device_through_cloud_path(self):
        privacy = importlib.import_module("privacy")
        self.assertTrue(privacy.is_allowed_cloud_lookup("weather in bengaluru"))
        self.assertFalse(privacy.is_allowed_cloud_lookup("explain my financial data"))
        self.assertFalse(privacy.can_send_audio(False))
        self.assertFalse(privacy.can_send_audio(True))

    def test_local_reasoning_has_fallback(self):
        policy = importlib.import_module("reasoning_policy")
        self.assertTrue(policy.should_fallback("I cannot answer this accurately"))
        self.assertFalse(policy.should_fallback("Here is the answer"))

    def test_knowledge_matches_are_readable_prose(self):
        knowledge = importlib.import_module("knowledge")
        records = {
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
        result = knowledge.find_relevant_records("income for John", records)
        self.assertIn("monthly income", result.casefold())
        self.assertIn("85000", result)
        self.assertNotIn("{", result)
        self.assertNotIn("[history]", result)


if __name__ == "__main__":
    unittest.main()
