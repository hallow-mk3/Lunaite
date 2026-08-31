<p align="center">
  <img src="docs/assets/lunaite_icon.png" alt="Lunaite Logo" width="130" style="border-radius: 20px;" />
</p>

<h1 align="center">Lunaite</h1>

<p align="center">
  <a href="https://github.com/hallow-mk3/Lunaite/releases"><img src="https://img.shields.io/github/v/release/hallow-mk3/Lunaite?color=06b6d4&label=Release" alt="Release" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-06b6d4.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-38bdf8.svg" alt="Python: 3.8+" /></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-Local%20%7C%20Cloud-38bdf8.svg" alt="Ollama Compatible" /></a>
  <a href="https://github.com/hallow-mk3/Lunaite/actions"><img src="https://img.shields.io/badge/Tests-Passing%20(16%2F16)-34d399.svg" alt="Tests" /></a>
</p>

<p align="center">
  <strong>An intelligent, lightweight tool harness and local AI agent runtime.</strong><br>
  <em>Connect any LLM, search only the tools you actually need, and run tasks directly from your terminal.</em>
</p>

---

## 💡 Why Lunaite?

When you build real AI assistants with 30+ tools, the default approach is to dump every single JSON schema into the model's prompt on every single turn.

That causes three big headaches:
1. **Token Waste & High Bills:** Repeating dozens of unused schemas wastes thousands of tokens per request.
2. **Context Window Contention:** Big tool lists crowd out your actual conversation history and files.
3. **Model Confusion (Distractor Overload):** When 40 tools look somewhat similar, models get distracted and pick the wrong one or invent weird parameters.

**Lunaite solves this in two ways:**
- **As a Library:** A clean tool harness that uses fast semantic search (`RetrievalSelector`) to pick only the top-$k$ relevant tools per turn. This cuts tool prompt tokens by up to **90%** while keeping accuracy high.
- **As an Agent CLI:** An interactive developer terminal (`python -m lunaite`) that auto-discovers your local Ollama models and cloud APIs, with live web search, memory compaction, session branching, and system tools.

---

## ⚡ Quick Start

### 1. Installation

Clone and install locally in editable mode:

```bash
git clone https://github.com/hallow-mk3/Lunaite.git
cd Lunaite
pip install -e .
```

*Optional extras (recommended for full semantic vector search):*
```bash
pip install sentence-transformers psutil pyautogui
```

---

### 2. Launch the Interactive Chat CLI

No complicated configs needed. Just run:

```bash
python -m lunaite
```

Lunaite will automatically scan your machine for local Ollama models (`qwen3`, `lunaite-ai`, `mistral`, etc.) and cloud API keys (`OPENAI_API_KEY`, `GROQ_API_KEY`), show an interactive selector, and start immediately.

```text
╭────────────────── Select AI Model Backend ───────────────────╮
  Local Models (Found on system):
    [1] lunaite-ai             Local Ollama (lunaite-ai)
    [2] qwen3:8b               Local Ollama (qwen3:8b)
    [3] qwen3.8:27b            Local Ollama (qwen3.8:27b)

  Cloud API Models (Key detected):
    [4] gpt-4o-mini            OpenAI (Fast, High Efficiency)
    [5] llama-3.3-70b          Groq (Ultra-Fast 70B)
╰──────────────────────────────────────────────────────────────╯
```

---

## 🛠️ Developer Slash Commands

Lunaite includes a rich set of developer commands right inside the chat session:

| Command | What it does |
|---|---|
| `/context` | Visualizes your current context usage as a colored percentage grid (Persona, Tools, Memory, History). |
| `/compact` | Summarizes older turns into dense episodic insights and frees up prompt tokens. |
| `/autocompact` | Sets how full the context gets before auto-compacting (e.g. `/autocompact 75%` or `/autocompact off`). |
| `/branch [name]` | Forks your current conversation into a named branch so you can explore alternative code/thoughts. |
| `/background` | Saves the entire session state to disk (`~/.lunaite/sessions/`) and frees the terminal immediately. |
| `/resume [id]` | Lists all saved sessions or restores a previous conversation by ID. |
| `/btw <query>` | Ask a quick side question without saving it to your conversation history or memory. |
| `/bug [desc]` | Exports a full diagnostic report (`lunaite_bug_*.json`) with system telemetry and recent transcript. |
| `/add-dir [path]` | Register additional working directories for multi-project development. |
| `/cd <path>` | Move the active session to a different directory. |
| `/clear` | Saves the current session to disk and starts a fresh conversation. |
| `/history` | View the conversation log for the active session. |
| `/deliberate` | Toggle multi-perspective cognitive reasoning (empirical logic, systems architecture, verification). |
| `/tools` | List all registered tools and their natural language trigger patterns. |
| `/info` | Show live CPU, RAM, GPU VRAM, and disk telemetry. |
| `/exit` | Auto-save session state and quit. |

---

## 🔍 Context Grid Visualization

Type `/context` anytime to see exactly where your tokens are going:

```text
╭───────────────────── Context Utilization ─────────────────────╮
  Context Buffer:  [████████░░░░░░░░░░░░░░░░] 2,450 / 8,192 tokens (29.9%)
  Auto-Compact:    Triggers at 80% (6,553 tokens)

  ▣ System Persona:    380 tok ( 4.6%)   ▣ Tools Registry:   450 tok ( 5.5%)
  ▣ Memory Bank:       180 tok ( 2.2%)   ▣ History Turns:  1,440 tok (17.6%)

  📁 Active Workspaces (2):
     1. C:\Users\Swasthik Shetty\AI (primary)
     2. C:\Users\Swasthik Shetty\AI\research
  🌿 Active Branch:    main  • Total Turns: 6
╰───────────────────────────────────────────────────────────────╯
```

---

## 🚀 30-Second Python Library Example

Use Lunaite inside your own scripts and applications:

```python
from lunaite.harness import Harness
from lunaite.tools import ToolRegistry
from lunaite.selection import RetrievalSelector

# 1. Create a registry and register functions
registry = ToolRegistry()

@registry.register(
    name="get_weather",
    description="Fetch current temperature and weather conditions for a city.",
    parameters={
        "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
    }
)
def get_weather(city: str, unit: str = "celsius"):
    return f"Weather in {city}: 22° {unit.upper()}, clear skies."

# 2. Pick a selector (Retrieval picks the top-k most relevant tools)
selector = RetrievalSelector(registry, k=3)

# 3. Create the harness (works with local Ollama or any OpenAI-compatible API)
harness = Harness(
    registry=registry,
    selector=selector,
    model="qwen3:8b",
    base_url="http://localhost:11434/v1"
)

# 4. Run your query
result = harness.run("What's the weather like in Tokyo right now?")

print("Tools shown to model:", result.tools_shown)
print("Tool called:         ", result.tool_called)
print("Result:              ", result.tool_output)
print("Prompt tokens used:  ", result.prompt_tokens)
```

---

## 🧠 Drop-in Agent Model Wrapper

Wrap any local or remote model with persistent memory, autonomous tool routing, and cognitive deliberation:

```python
import lunaite

# Wrap any model (Ollama, HuggingFace, OpenAI, Groq)
model = lunaite.wrap("qwen3:8b")

# Generate with autonomous real-time tools & memory
response = model.generate("When did the James Webb Space Telescope launch?", use_agent=True)
print(response)

# Multi-perspective deep deliberation
analysis = model.generate("Compare Raft vs Paxos consensus in distributed systems.", use_deliberation=True)
print(analysis)
```

---

## 📊 Research & Benchmark Suite

Lunaite includes a reproducible tool selection benchmark under [`research/tool_selection/`](research/tool_selection/):

- **150 Curated Benchmark Tasks:** Unambiguous queries, confusable tools (e.g. `convert_currency` vs `get_exchange_rate`), and trick/negative cases.
- **Run the evaluation:**
  ```bash
  # Evaluate Naive vs Retrieval (k=3, 5, 10) across scale
  python research/tool_selection/run_eval.py --model qwen3:8b

  # Generate analysis charts & summary tables
  python research/tool_selection/analyze.py
  ```

Read the full evaluation report in [`research/tool_selection/REPORT.md`](research/tool_selection/REPORT.md).

---

## 📁 Repository Structure

```
Lunaite/
├── lunaite/
│   ├── harness/             # Tool execution harness, logging & validation
│   ├── selection/           # Naive vs. dense vector Retrieval selectors
│   ├── tools/               # Tool registry, schemas & decorator engine
│   ├── agent/               # Real-time search, wikipedia, weather & desktop tools
│   ├── core/                # Multi-tier persistent memory & cognitive deliberation
│   ├── models/              # Unified model wrapper (Ollama, API, HuggingFace)
│   └── cli.py               # Modern interactive terminal runtime
├── research/
│   └── tool_selection/      # 150-task benchmark suite, runner & report
├── tests/                   # Unit test suite (16 test cases)
└── pyproject.toml           # Package configuration
```

---

## 🧪 Testing

Run the full unit test suite:

```bash
python -m unittest discover tests
```

---

## 📄 License

Lunaite is open source under the [MIT License](LICENSE).

Developed and maintained with care by **[Swasthik Shetty](https://github.com/hallow-mk3)** ([swasthik.mk3@gmail.com](mailto:swasthik.mk3@gmail.com)).
