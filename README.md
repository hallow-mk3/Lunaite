# Lunaite

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-3776ab.svg)](https://www.python.org/)
[![Ollama Compatible](https://img.shields.io/badge/Ollama-Native-black.svg)](https://ollama.com/)
[![Code Style: Clean](https://img.shields.io/badge/Code%20Style-Black%20%2F%20Typed-informational.svg)](https://github.com/hallow-mk3/Lunaite)

**A lightweight, modular tool-selection harness and agent toolkit for LLMs.**

Most tool-augmented LLM systems have a dirty secret: when you give a model 30+ tools, you usually just dump all 30 JSON schemas into the prompt and pray. As your tool registry grows, prompts get bloated, token costs spike, and the model starts hallucinating arguments or picking the wrong function altogether.

**Lunaite** was built to solve and study this exact problem. It provides:
1. **A clean tool-access harness** — Standardized schemas, robust function-call parsing, execution safety, and step-by-step telemetry.
2. **Intelligent tool selection** — Compare **Naive** (all schemas in context) vs. **Retrieval-based** (dense semantic top-$k$ tool selection) before calling the model.
3. **An empirical research suite** — 150+ synthetic benchmark tasks (unambiguous, confusable/ambiguous, and trick/negative cases) across 50+ diverse tools to measure accuracy, argument validity, and token savings.
4. **An interactive agent & CLI** — Chat with local Ollama models, Hugging Face checkpoints, or cloud APIs with persistent memory and tools right in your terminal.

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/hallow-mk3/Lunaite.git
cd Lunaite
pip install -e .
```

*Optional extras:*
```bash
# For dense vector retrieval tool selection:
pip install sentence-transformers

# For research plotting:
pip install matplotlib numpy
```

---

## 🛠️ The Core Concept: Smarter Tool Selection

Instead of force-feeding 50 function schemas to an LLM on every single query, Lunaite filters down to the most relevant tools first:

```
                      ┌────────────────────────────────────────┐
                      │ User Query: "Convert 100 USD to EUR"   │
                      └──────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
      [ Naive Selection ]                             [ Retrieval Selection ]
  Dumps all 50+ tool schemas                      Embeds query & tool descriptions
  into the prompt context                         Picks only Top-k relevant tools (e.g. k=3)
                 │                                               │
                 ▼                                               ▼
  ~4,200 prompt tokens                            ~420 prompt tokens (~90% savings)
  Higher chance of confusion                      Laser-focused context
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                             ┌───────────────────────┐
                             │ LLM Generates Call    │
                             │ Executes & Validates  │
                             └───────────────────────┘
```

---

## 🚀 30-Second Python Example

Here is how simple it is to build a registry, choose a selection strategy, and run queries through the harness:

```python
from lunaite.harness import Harness
from lunaite.tools import Tool, ToolRegistry
from lunaite.selection import RetrievalSelector, NaiveSelector

# 1. Create a registry and register your tools
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

# 2. Pick a selector (Retrieval uses sentence embeddings; falls back to TF-IDF)
selector = RetrievalSelector(registry, k=3)

# 3. Spin up the harness (connects to local Ollama or any OpenAI-compatible API)
harness = Harness(
    registry=registry,
    selector=selector,
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1"
)

# 4. Run your query
result = harness.run("What's the weather like in Tokyo right now?")

print(f"Tools shown to LLM: {result.tools_shown}")
print(f"Tool chosen:        {result.tool_called}")
print(f"Arguments:          {result.tool_args}")
print(f"Tool Output:        {result.tool_output}")
print(f"Prompt Tokens:      {result.prompt_tokens}")
```

---

## 📊 Research: Benchmarking Tool Selection at Scale

Lunaite includes a reproducible research evaluation suite under [`research/tool_selection`](research/tool_selection/) designed to test model performance across varying tool library sizes (10 to 50+ tools).

### The Benchmark Dataset (`eval_tasks.jsonl`)
- **150 curated tasks** across 10 functional domains (weather, math, finance, geocoding, strings, dates, system metrics, unit conversion, wikipedia, etc.).
- **Unambiguous cases**: Query maps directly to one clear tool.
- **Ambiguous & confusable cases**: Query presents overlapping options (e.g., `convert_currency` vs `get_exchange_rate`, `calc_hypotenuse` vs `calc_distance_2d`) to test precision.
- **Trick / Negative cases**: Query where no registered tool should be used, testing the model's discipline to reply without forcing a hallucinated tool call.

### Running the Eval

```bash
# 1. Run the benchmark across Naive and Retrieval (k=3, 5, 10)
python research/tool_selection/run_eval.py --model qwen2.5:7b

# 2. Generate summary tables and publication-ready dark-theme plots
python research/tool_selection/analyze.py
```

### Key Metrics Tracked
- **Tool Selection Accuracy**: Did the model pick the ground-truth tool?
- **Argument Correctness**: Were parameter names and values valid?
- **Full Success Rate**: Correct tool selection + valid arguments + error-free execution.
- **Prompt Token Footprint**: Total context overhead saved.
- **End-to-End Latency**: Time to select, generate, and execute.

---

## 💻 Interactive Terminal CLI

Prefer chatting directly with tools in your terminal? Use Lunaite's built-in interactive CLI:

```bash
# Chat with any Ollama model with autonomous tool calling
python -m lunaite.cli run qwen2.5:7b

# Run with multi-perspective deliberation & verification
python -m lunaite.cli run qwen2.5:7b --deliberate
```

Inside the chat:
- Type `/tools` to see all available tools in real time.
- Type `/memory` to view what the agent has remembered across sessions.
- Type `/clear` to reset active context.

---

## 🧠 High-Level Model Wrapper

If you want an drop-in intelligence layer around standard LLM clients:

```python
import lunaite

# Wrap any local or remote model
model = lunaite.wrap("qwen2.5:7b")

# Generate with autonomous tool reasoning
response = model.generate("What is the current time in Tokyo and the weather?", use_agent=True)
print(response)

# Multi-step deliberation (Logic, Physics, Systems Analysis)
analysis = model.generate("How does Raft consensus handle network partitions?", use_deliberation=True)
print(analysis)
```

---

## 📁 Repository Structure

```
Lunaite/
├── lunaite/
│   ├── harness/             # Core LLM tool harness & execution logger
│   │   └── harness.py       # Orchestration loop, parsing, and execution
│   ├── selection/           # Tool selection algorithms
│   │   ├── base.py          # Selector abstract base class
│   │   ├── naive.py         # Pass-all-tools baseline
│   │   └── retrieval.py     # Dense vector top-k retrieval (with TF-IDF fallback)
│   ├── tools/               # Tool registry & schemas
│   │   ├── registry.py      # Registration decorators & schema validation
│   │   └── tool.py          # Tool dataclass & JSON schema generator
│   ├── agent/               # Autonomous agent loop, tools, and telemetry
│   ├── core/                # Memory persistence and multi-expert reasoning
│   ├── models/              # Model backends (Ollama, HuggingFace, OpenAI-compatible)
│   └── cli.py               # Interactive terminal interface
├── research/
│   └── tool_selection/      # Scaled tool selection research
│       ├── tools_library.py # 50+ varied and confusable tools
│       ├── eval_tasks.jsonl # 150 benchmark test cases
│       ├── gen_eval_tasks.py# Task generator & validator
│       ├── run_eval.py      # Async-friendly benchmark runner
│       └── analyze.py       # Metrics aggregator and chart generator
├── docs/                    # Architecture diagrams & documentation
├── tests/                   # Unit & integration test suite
└── pyproject.toml           # Package configuration
```

---

## 🛡️ Deliberate Design & Safety Stance

- **Read-Only / Safe Defaults**: The built-in evaluation and standard tool registries explicitly focus on safe, non-destructive tools (math, conversion, search, calculations, time). Destructive filesystem, arbitrary shell execution, and clipboard access are isolated to protect developer environments.
- **Zero Heavy Bloat**: The core harness has virtually zero hard external dependencies—it runs on standard Python 3.8+ with standard library / `requests` and gracefully leverages `sentence-transformers` and `matplotlib` when available.
- **Structured Telemetry**: Every execution step produces a typed, JSON-serializable `HarnessResult` for debugging, evals, or streaming down to web clients.

---

## 🤝 Contributing

Contributions, bug reports, and new tool-selection ideas (e.g. rerankers, hierarchical clustering, LLM-based pre-selectors) are very welcome!

```bash
# Run unit tests
python -m unittest discover tests
```

---

## 📄 License

Lunaite is licensed under the [MIT License](LICENSE).

Developed and maintained with care by **[Swasthik Shetty](https://github.com/hallow-mk3)** ([swasthik.mk3@gmail.com](mailto:swasthik.mk3@gmail.com)).
