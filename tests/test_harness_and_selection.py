import unittest
import json
from pathlib import Path

from lunaite.tools import Tool, ToolRegistry
from lunaite.selection import NaiveSelector, RetrievalSelector, HybridSelector
from lunaite.harness.harness import Harness, HarnessResult
from research.tool_selection.tools_library import build_registry, ALL_TOOLS


class TestHarnessAndSelection(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()

        @self.registry.register(
            name="get_weather",
            description="Fetch weather for a city",
            parameters={"city": {"type": "string"}}
        )
        def get_weather(city: str):
            return f"Sunny in {city}"

        @self.registry.register(
            name="convert_currency",
            description="Convert amount between two currencies",
            parameters={
                "amount": {"type": "number"},
                "from_curr": {"type": "string"},
                "to_curr": {"type": "string"}
            }
        )
        def convert_currency(amount: float, from_curr: str, to_curr: str):
            return f"{amount} {from_curr} = {amount * 1.1} {to_curr}"

        @self.registry.register(
            name="calculate_sum",
            description="Add two numbers together",
            parameters={"a": {"type": "number"}, "b": {"type": "number"}}
        )
        def calculate_sum(a: float, b: float):
            return a + b

    def test_registry_registration_and_retrieval(self):
        self.assertEqual(len(self.registry), 3)
        self.assertTrue("get_weather" in self.registry)
        tool = self.registry.get("get_weather")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "get_weather")
        self.assertEqual(tool("Tokyo"), "Sunny in Tokyo")

    def test_naive_selector(self):
        selector = NaiveSelector(self.registry)
        selected = selector.select("What is 5 plus 10?")
        self.assertEqual(len(selected), 3)
        names = [t.name for t in selected]
        self.assertIn("get_weather", names)
        self.assertIn("convert_currency", names)
        self.assertIn("calculate_sum", names)

    def test_retrieval_selector(self):
        selector = RetrievalSelector(self.registry, k=1)
        selected_weather = selector.select("Is it raining in London today?")
        self.assertEqual(len(selected_weather), 1)
        self.assertEqual(selected_weather[0].name, "get_weather")

        selected_math = selector.select("Calculate the sum of two values")
        self.assertEqual(len(selected_math), 1)
        self.assertEqual(selected_math[0].name, "calculate_sum")

    def test_retrieval_selector_k_greater_than_total(self):
        selector = RetrievalSelector(self.registry, k=10)
        selected = selector.select("Any query")
        self.assertEqual(len(selected), 3)

    def test_hybrid_selector(self):
        selector = HybridSelector(self.registry, k=1, alpha=0.5)
        selected_weather = selector.select("Fetch current weather in Tokyo")
        self.assertEqual(len(selected_weather), 1)
        self.assertEqual(selected_weather[0].name, "get_weather")

        selected_curr = selector.select("Convert currency USD to EUR")
        self.assertEqual(len(selected_curr), 1)
        self.assertEqual(selected_curr[0].name, "convert_currency")

    def test_harness_openai_schema_conversion(self):
        tool = self.registry.get("get_weather")
        schema = tool.to_openai_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "get_weather")
        self.assertIn("parameters", schema["function"])

    def test_tools_library_integrity(self):
        reg = build_registry(50)
        self.assertGreaterEqual(len(reg), 50)
        for tool in ALL_TOOLS:
            self.assertIn(tool.name, reg)
            self.assertTrue(tool.description)
            self.assertIsInstance(tool.parameters, dict)

    def test_eval_tasks_jsonl_validity(self):
        tasks_file = Path("research/tool_selection/eval_tasks.jsonl")
        self.assertTrue(tasks_file.exists())
        with open(tasks_file, encoding="utf-8") as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        self.assertGreaterEqual(len(tasks), 100)
        for t in tasks:
            self.assertIn("id", t)
            self.assertIn("query", t)
            self.assertIn("case_type", t)
            self.assertIn("correct_tool", t)


if __name__ == "__main__":
    unittest.main()
