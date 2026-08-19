"""
Unit tests for Lunaite Memory Subsystem.
"""

import os
import unittest
import tempfile
from lunaite.config import MemoryConfig
from lunaite.core.memory import LunaiteMemory


class TestLunaiteMemory(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.config = MemoryConfig(filepath=self.temp_file.name, auto_persist=True)
        self.memory = LunaiteMemory(self.config)

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_remember_and_get(self):
        self.memory.remember("user_facts", "user_city", "Bangalore")
        self.assertEqual(self.memory.get_fact("user_city"), "Bangalore")

    def test_episodic_insights(self):
        self.memory.add_insight("User is conducting astrophysics research.")
        summary = self.memory.get_context_summary()
        self.assertIn("astrophysics research", summary)
        self.assertIn("Swasthik Shetty", summary)


if __name__ == "__main__":
    unittest.main()
