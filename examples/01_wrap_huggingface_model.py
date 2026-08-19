"""
Example 1: Wrapping ANY Hugging Face Model with Lunaite MoE Architecture
========================================================================
This example demonstrates how to load any Hugging Face model and attach
the Lunaite Sparse MoE Adapter to it.
"""

import lunaite

# 1. Attach Lunaite Architecture to any model (e.g. Qwen, LLaMA, Mistral, Gemma)
print("Attaching Lunaite Architecture to HuggingFace model...")
model = lunaite.wrap("Qwen/Qwen2.5-1.5B-Instruct", backend="huggingface")

# 2. Run inference with multi-tier memory and persona calibration
prompt = "Explain why quantum entanglement does not violate special relativity."
response = model.generate(prompt)

print(f"\nResponse:\n{response}")
