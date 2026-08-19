"""
Unit tests for Lunaite Agent Tools and Telemetry.
"""

import unittest
from lunaite.agent.suite import LunaiteAgent
from lunaite.agent.desktop import get_system_telemetry


class TestLunaiteTools(unittest.TestCase):

    def test_system_telemetry(self):
        telem = get_system_telemetry()
        self.assertIn("platform", telem)
        self.assertIn("ram_percent", telem)
        self.assertIn("cpu_percent", telem)

    def test_agent_intent_detection(self):
        agent = LunaiteAgent()
        # Weather detection
        intent = agent.detect_intent("What is the weather in Tokyo?")
        self.assertIsNotNone(intent)
        self.assertEqual(intent[0], "weather")

        # URL detection
        intent = agent.detect_intent("Can you summarize https://example.com?")
        self.assertIsNotNone(intent)
        self.assertEqual(intent[0], "fetch_url")

        # Telemetry detection
        intent = agent.detect_intent("Show me current hardware stats and system vitals")
        self.assertIsNotNone(intent)
        self.assertEqual(intent[0], "telemetry")


if __name__ == "__main__":
    unittest.main()
