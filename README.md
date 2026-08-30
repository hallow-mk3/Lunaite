# Lunaite

**Give any AI model multi-step reasoning, persistent memory, live web tools, and MoE routing.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Quick Start

```bash
pip install -e .
```

### Wrap an Ollama model in 3 lines:

```python
import lunaite

# Connect to any local model
model = lunaite.from_ollama("qwen2.5:7b")

# Use live web tools
print(model.generate("What is the latest news on NASA missions?", use_agent=True))

# Multi-perspective reasoning (Physics, Logic, Philosophy)
print(model.generate("Explain quantum superposition and its core paradoxes.", use_deliberation=True))
```

---

## Features & Usage

### 1. Universal Model Wrapping
Works seamlessly with Ollama, Hugging Face, or OpenAI-compatible APIs:

```python
import lunaite

m1 = lunaite.wrap("llama3.1:8b")                    # Local Ollama
m2 = lunaite.wrap("meta-llama/Llama-3-8B-Instruct")  # Hugging Face
m3 = lunaite.wrap("gpt-4o", backend="api")           # Cloud API

# Remembers past conversation facts
print(m1.chat("Remember that my project is called Lunaite."))
```

### 2. Standalone PyTorch MoE Layer
Add sparse Mixture-of-Experts routing directly into your PyTorch models:

```python
import torch
from lunaite import LunaiteMoELayer

moe = LunaiteMoELayer(d_model=4096, num_experts=8, top_k=2, expert_dim=1024)
x = torch.randn(2, 16, 4096)
output, aux_loss = moe(x)
```

---

## Command Line Interface

```bash
# Quick terminal chat
lunaite run qwen2.5:7b --deliberate

# Train LoRA / MoE adapters
lunaite train --base-model Qwen/Qwen2.5-7B --dataset data/lunaite_training_data.jsonl --epochs 3
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

