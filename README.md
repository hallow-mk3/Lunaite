<p align="center">
  <img src="docs/assets/lunaite_icon.png" alt="Lunaite Logo" width="140" style="border-radius: 24px;" />
</p>

<h1 align="center">Lunaite</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-3776ab.svg" alt="Python: 3.8+" /></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-Native-black.svg" alt="Ollama Compatible" /></a>
</p>

<p align="center">
  <strong>Smarter tool selection and a clean, lightweight agent toolkit for LLMs.</strong>
</p>

---

### Why Lunaite?

When you give an LLM 30+ tools, the default approach is usually to dump every single JSON schema into the system prompt. 

As your tool library grows, this causes three big problems:
1. **Token bills skyrocket** from repeating dozens of unused schemas on every call.
2. **Context windows get crowded**, leaving less room for user history and data.
3. **Models get confused**, frequently picking the wrong tool or hallucinating parameters.

**Lunaite** fixes this by retrieving and injecting **only the tools relevant to the prompt** before calling the model — saving up to **90% of your tool-related tokens** while boosting tool selection accuracy.

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/hallow-mk3/Lunaite.git
cd Lunaite
pip install -e .
```

*Optional extras:*
```bash
# For dense semantic vector search (recommended):
pip install sentence-transformers

# For benchmark plots & visual analytics:
pip install matplotlib numpy
```

---

## 💡 How It Works

Instead of passing all 50+ tools in the prompt, Lunaite filters down to the top-$k$ most relevant tools first:

- **Naive (Baseline):** Passes all registered tools in the prompt context (~4,000+ tokens, higher confusion).
- **Retrieval (Lunaite):** Embeds the query and picks the top-$k$ relevant tools (e.g. $k=3$, ~400 tokens, laser-focused).

---

## 🚀 30-Second Example

Register tools with a simple decorator, choose a selector, and run:

```python
from lunaite.harness import Harness
from lunaite.tools import ToolRegistry
from lunaite.selection import RetrievalSelector

# 1. Create a registry and define your tools
registry = ToolRegistry()

@registry.register(
    name="get_current_weather",
    description="Fetch current temperature and weather conditions for a given city.",
    parameters={
        "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
    }
)
def get_weather(city: str, unit: str = "celsius"):
    return f"Weather in {city}: 22° {unit.upper()}, clear skies."

# 2. Pick a selector (Retrieval uses embeddings with TF-IDF fallback)
selector = RetrievalSelector(registry, k=3)

# 3. Spin up the harness (works with Ollama or any OpenAI-compatible endpoint)
harness = Harness(
    registry=registry,
    selector=selector,
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1"
)

# 4. Run the query
result = harness.run("What's the weather like in Tokyo right now?")

print("Tools shown to LLM:", result.tools_shown)
print("Tool executed:     ", result.tool_called)
print("Output:            ", result.tool_output)
print("Tokens used:       ", result.prompt_tokens)
```

---

## 🧠 High-Level Agent Wrapper

If you want a drop-in agent wrapper around your favorite model:

```python
import lunaite

# Wrap any local or remote model
model = lunaite.wrap("qwen2.5:7b")

# Generate with autonomous tool reasoning
response = model.generate("What is the current time in Tokyo and the weather?", use_agent=True)
print(response)

# Multi-perspective deliberation (logic, systems, verification)
analysis = model.generate("How does Raft consensus handle network partitions?", use_deliberation=True)
print(analysis)
```

---

## 💻 Interactive Terminal CLI

Chat directly with local models and test your tools right in the terminal:

```bash
# Chat with any Ollama model with tool calling enabled
python -m lunaite.cli run qwen2.5:7b

# Run with multi-perspective deliberation & verification
python -m lunaite.cli run qwen2.5:7b --deliberate
```

**Commands inside chat:**
- `/tools` — List all currently active tools.
- `/memory` — Inspect agent memory across sessions.
- `/clear` — Reset session history.

---

## 📊 Research & Benchmarks

Lunaite includes a reproducible evaluation suite under [`research/tool_selection/`](research/tool_selection/) to test tool selection accuracy and latency at scale (10 to 50+ tools).

- **150 Curated Tasks:** Direct queries, confusable tools (e.g. `convert_currency` vs `get_exchange_rate`), and trick/negative cases.
- **Run the eval:**
  ```bash
  # Evaluate Naive vs Retrieval (k=3, 5, 10)
  python research/tool_selection/run_eval.py --model qwen2.5:7b

  # Generate analysis & plots
  python research/tool_selection/analyze.py
  ```

---

## 📁 Project Structure

```
Lunaite/
├── lunaite/
│   ├── harness/             # LLM orchestration loop, parsing, and execution
│   ├── selection/           # Naive vs. dense vector Retrieval selectors
│   ├── tools/               # Tool registry, schemas, and decorators
│   ├── agent/               # Autonomous agent loop & telemetry
│   ├── core/                # Persistent memory & multi-expert deliberation
│   ├── models/              # Backends (Ollama, HuggingFace, OpenAI-compatible)
│   └── cli.py               # Interactive terminal interface
├── research/
│   └── tool_selection/      # 150-task benchmark suite & visualization scripts
├── tests/                   # Unit and integration tests
└── pyproject.toml           # Package configuration
```

---

## 🛡️ Key Principles

- **Safe Defaults**: Built-in evaluation tools are non-destructive (calculations, conversions, lookups).
- **Lightweight**: Zero heavy required dependencies — works out of the box with standard Python and adds vector retrieval when available.
- **Typed Telemetry**: Every call produces a clean `HarnessResult` object with execution logs, token counts, and argument validation.

## 🤝 Contributing & License

Contributions, bug reports, and new tool selection strategies (rerankers, hierarchical clustering, etc.) are always welcome!

```bash
# Run tests
python -m unittest discover tests
```

---

## 📄 License

Lunaite is licensed under the [MIT License](LICENSE).

Developed and maintained with care by **[Swasthik Shetty](https://github.com/hallow-mk3)** ([swasthik.mk3@gmail.com](mailto:swasthik.mk3@gmail.com)).
