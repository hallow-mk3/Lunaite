# Lunaite Tutorial & Complete Guide

**A practical walkthrough for using Lunaite to give any model reasoning, memory, tools, MoE layers, and interactive diagrams.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Table of Contents

1. [Installation](#1-installation)
2. [Wrapping Models (Ollama, Hugging Face, APIs)](#2-wrapping-models)
3. [Deliberation & Reasoning](#3-deliberation--reasoning)
4. [Live Web Tools & Telemetry](#4-live-web-tools--telemetry)
5. [Persistent Memory Bank](#5-persistent-memory-bank)
6. [PyTorch MoE Layers](#6-pytorch-moe-layers)
7. [Interactive Diagrams Visualizer](#7-interactive-diagrams-visualizer)
8. [Command Line Interface (CLI)](#8-command-line-interface-cli)
9. [Fine-Tuning & Adapter Export](#9-fine-tuning--adapter-export)

---

## 1. Installation

Install in editable mode for development:

```bash
git clone https://github.com/hallow-mk3/Lunaite.git
cd Lunaite
pip install -e .
```

---

## 2. Wrapping Models

Lunaite's `wrap()` function automatically detects your backend and equips the model with cognitive memory and tool capabilities.

### Local Ollama Models

```python
import lunaite

# Connect to any local model (e.g. Qwen 2.5, LLaMA 3.1, Mistral)
model = lunaite.wrap("qwen2.5:7b")

response = model.chat("Hello! What are you capable of?")
print(response)
```

### Hugging Face Transformers

```python
import lunaite

# Wrap a Hugging Face CausalLM
model = lunaite.wrap("Qwen/Qwen2.5-1.5B-Instruct", backend="huggingface")
print(model.chat("Explain the difference between TCP and UDP."))
```

### Cloud APIs (OpenAI-compatible)

```python
import lunaite

# Wrap an API model
model = lunaite.wrap("gpt-4o", backend="api")
print(model.chat("Summarize today's goals."))
```

---

## 3. Deliberation & Reasoning

When you need deep, multi-perspective analysis on complex questions, enable deliberation:

```python
import lunaite

model = lunaite.from_ollama("qwen2.5:7b")

# Runs parallel perspective analysis (Physics, Logic, Systems) and verifies the answer
deep_answer = model.generate(
    "Explain the implications of quantum gravity on black hole information loss.",
    use_deliberation=True
)
print(deep_answer)
```

---

## 4. Live Web Tools & Telemetry

Lunaite includes autonomous tool detection for live web search, Wikipedia lookups, weather forecasts, URL article scraping, and system telemetry:

```python
import lunaite

model = lunaite.from_ollama("qwen2.5:7b")

# Live web search & synthesis
print(model.generate("What is the latest news on Artemis missions?", use_agent=True))

# Live weather
print(model.generate("What is the weather in Tokyo right now?", use_agent=True))

# System hardware status
print(model.generate("What are my current system vitals and RAM usage?", use_agent=True))
```

---

## 5. Persistent Memory Bank

Lunaite saves user facts, preferences, and key conversation insights across sessions into `lunaite_memory.json`:

```python
import lunaite

model = lunaite.from_ollama("qwen2.5:7b")

# Store a fact
model.chat("Please remember that my favorite programming language is Rust.")

# In a later session, Lunaite automatically recalls this context
print(model.chat("What programming language should I use for a high-throughput network service?"))
```

You can also access memory directly:

```python
from lunaite import LunaiteMemory

memory = LunaiteMemory()
memory.remember("user_facts", "location", "Bengaluru")
print(memory.get_fact("location"))  # 'Bengaluru'
```

---

## 6. PyTorch MoE Layers

You can use Lunaite's sparse Mixture-of-Experts building blocks directly inside your own neural networks:

```python
import torch
from lunaite import LunaiteMoELayer

# 8 experts, top-2 active per token
moe = LunaiteMoELayer(
    d_model=4096,
    num_experts=8,
    top_k=2,
    expert_dim=1024
)

# Input shape: (batch_size, seq_len, d_model)
tokens = torch.randn(2, 16, 4096)
output, aux_loss = moe(tokens)

print("Output tensor shape:", output.shape)
print("Load balancing loss:", aux_loss.item())
```

---

## 7. Interactive Diagrams Visualizer

Generate vector-crisp, dark-themed interactive HTML architecture and sequence diagrams with zoom controls and one-click SVG export:

```bash
# Render a diagram to standalone HTML
python .agents/skills/interactive-diagrams/scripts/render_diagram.py \
  --title "System Architecture" \
  --output "docs/my_diagram.html"
```

Or programmatically in Python:

```python
from .agents.skills.interactive_diagrams.scripts.render_diagram import generate_diagram_html

mermaid_code = """
graph TD
    Client[Client Request] --> Engine[Lunaite Engine]
    Engine --> Deliberation[Cognitive Deliberation]
    Engine --> Tools[Live Web Tools]
    Engine --> MoE[MoE Router]
    MoE --> Output[Consensus Output]
"""

generate_diagram_html(
    title="Workflow Topology",
    description="Lunaite Pipeline Overview",
    mermaid_code=mermaid_code,
    output_path="docs/pipeline.html"
)
```

---

## 8. Command Line Interface (CLI)

Interact directly from your terminal:

```bash
# Terminal chat with any model
lunaite run qwen2.5:7b
# or via python module directly:
python -m lunaite.cli run qwen2.5:7b

# Terminal chat with deep deliberation
lunaite run llama3.1:8b --deliberate
# or:
python -m lunaite.cli run llama3.1:8b --deliberate

# View hardware stats and diagnostics
lunaite info
# or:
python -m lunaite.cli info
```

---

## 9. Fine-Tuning & Adapter Export

Fine-tune models with LoRA and MoE adapters and merge weights:

```bash
# Train on a dataset
lunaite train --base-model Qwen/Qwen2.5-7B --dataset data/lunaite_training_data.jsonl --epochs 3

# Merge adapter weights into a standalone model
lunaite merge --base-model Qwen/Qwen2.5-7B --adapter ./lunaite_weights --output ./lunaite_merged
```

---

## Testing

Run the automated test suite:

```bash
python -m unittest discover tests
```

---

## License & Author

- **Author:** [Swasthik Shetty](https://github.com/hallow-mk3) ([swasthik.mk3@gmail.com](mailto:swasthik.mk3@gmail.com))
- **License:** [MIT License](LICENSE)
