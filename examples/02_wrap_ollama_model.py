"""
Example 2: Wrapping ANY Local Ollama Model with Lunaite Architecture
====================================================================
This example demonstrates how to empower any locally running Ollama model
(Qwen, LLaMA 3, Mistral, Gemma, Phi, DeepSeek) with Lunaite Cognitive Deliberation,
multi-tier persistent memory, and autonomous internet tools.
"""

import lunaite

# 1. Connect to any model served by Ollama (or 'lunaite-ai')
model = lunaite.from_ollama("qwen2.5:7b")

# 2. Standard Generation with Autonomous Agent Tools & Live Web Search
print("--- 1. Autonomous Live Information Query ---")
query_1 = "What is the weather in Tokyo today and what are recent space exploration milestones?"
resp_1 = model.generate(query_1, use_agent=True)
print(resp_1)

# 3. Multi-Perspective Cognitive Deliberation (Debate & Synthesis)
print("\n--- 2. Multi-Perspective Cognitive Deliberation ---")
query_2 = "What are the physical and philosophical limits of artificial general intelligence?"
resp_2 = model.generate(query_2, use_deliberation=True)
print(resp_2)
