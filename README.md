# Lunaite

**Universal Cognitive Architecture & Agent Runtime for ANY AI Model.**

Lunaite attaches multi-step reasoning, persistent memory, live web search, system telemetry, and MoE routing to your local models (Ollama), Hugging Face models, or Cloud APIs—just like a modular intelligence layer.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/hallow-mk3/Lunaite.git
cd Lunaite
pip install -e .
```

### 1. Terminal Chat (CLI)

Run interactive chat with any local Ollama model (auto-launches Ollama if not already running):

```bash
# Standard chat with tools (DuckDuckGo, Wikipedia, Weather, Telemetry)
python -m lunaite.cli run qwen2.5:7b

# Chat with deep multi-perspective deliberation & verification
python -m lunaite.cli run qwen2.5:7b --deliberate
```

---

### 2. Python SDK

```python
import lunaite

# Wrap any local Ollama model
model = lunaite.wrap("qwen2.5:7b")

# Use live web tools automatically
print(model.generate("What is the latest news on NASA missions?", use_agent=True))

# Multi-perspective reasoning (Physics, Logic, Practical Systems)
print(model.generate("Explain quantum superposition and its core paradoxes.", use_deliberation=True))

# Remembers conversation context across sessions
print(model.chat("Remember that my primary development language is Python."))
```

---

## Supported Backends

Lunaite seamlessly wraps any model backend:

```python
import lunaite

# 1. Local Ollama (llama3, qwen2.5, mistral, gemma, phi3)
m1 = lunaite.wrap("llama3.1:8b")

# 2. Hugging Face Transformers
m2 = lunaite.wrap("meta-llama/Llama-3-8B-Instruct", backend="huggingface")

# 3. Cloud APIs (OpenAI-compatible)
m3 = lunaite.wrap("gpt-4o", backend="api")
```

---

## Standalone PyTorch MoE Layer

You can also use Lunaite's sparse Mixture-of-Experts building blocks inside your own neural architectures:

```python
import torch
from lunaite import LunaiteMoELayer

moe = LunaiteMoELayer(d_model=4096, num_experts=8, top_k=2, expert_dim=1024)
x = torch.randn(2, 16, 4096)
output, aux_loss = moe(x)
```

---

## Interactive Diagrams

Generate vector-crisp standalone HTML diagrams for architectures and workflows:

```bash
python .agents/skills/interactive-diagrams/scripts/render_diagram.py \
  --title "System Architecture" \
  --output "docs/architecture.html"
```

---

## Testing

```bash
python -m unittest discover tests
```

---

## License & Author

- **Author:** [Swasthik Shetty](https://github.com/hallow-mk3) ([swasthik.mk3@gmail.com](mailto:swasthik.mk3@gmail.com))
- **License:** [MIT License](LICENSE)
