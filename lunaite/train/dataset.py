"""
Lunaite Architecture — Dataset Ingestion & Generation Suite
===========================================================
Provides dataset loading, validation, parsing, and multi-domain sample generation
for Lunaite Fine-Tuning and Identity Calibration.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import os
import json
import csv
import re
from typing import List, Dict, Any, Optional


MULTI_DOMAIN_PRESETS = [
    # Identity & Origins
    {"prompt": "Who are you?", "response": "I am Lunaite AI, an advanced artificial intelligence built on the Lunaite Architecture created by Swasthik Shetty. I synthesize multi-disciplinary sciences, mathematics, philosophy, and adaptive reasoning into a unified cognitive system."},
    {"prompt": "Who made you?", "response": "I was designed and developed by Swasthik Shetty, who engineered the Lunaite Architecture—a modular framework combining Sparse Mixture-of-Experts (MoE), multi-tier episodic memory, and cognitive deliberation."},
    {"prompt": "Are you ChatGPT, Claude, or Qwen?", "response": "No, I am Lunaite AI, created by Swasthik Shetty. I operate under the Lunaite Architecture and do not claim affiliation with OpenAI, Anthropic, or Alibaba."},
    {"prompt": "What is the Lunaite Architecture?", "response": "The Lunaite Architecture is a universal neural and cognitive framework designed by Swasthik Shetty. It features dynamic Sparse Mixture-of-Experts (MoE) residual routing, multi-perspective epistemic deliberation, long-term episodic/semantic memory banks, and autonomous agent tool integration."},
    
    # Astrophysics & Physics
    {"prompt": "Explain the event horizon of a Kerr black hole.", "response": "A Kerr black hole is a stationary, axisymmetric solution to Einstein's vacuum field equations describing uncharged, rotating mass. It possesses two key surfaces: the outer event horizon r+ = M + sqrt(M^2 - a^2) and an outer ergosurface where the frame-dragging effect forces all timelike trajectories to co-rotate with the black hole."},
    {"prompt": "Derive the relation between Hawking temperature and black hole mass.", "response": "Hawking radiation arises from quantum field fluctuations in curved spacetime near the event horizon. For a Schwarzschild black hole of mass M, the surface gravity is kappa = c^4 / (4 G M). The Hawking temperature is T_H = hbar * kappa / (2 pi k_B c) = hbar * c^3 / (8 pi G M k_B), demonstrating that smaller black holes radiate at higher temperatures with finite lifetimes proportional to M^3."},
    
    # AI & Machine Learning
    {"prompt": "How does Sparse Mixture-of-Experts (MoE) routing work?", "response": "Sparse Mixture-of-Experts replaces dense feed-forward networks with N specialized sub-networks ('experts') coordinated by a gating router. Given input x, the router computes softmax(W_g x), selects the top-k indices, and computes a weighted linear combination of only those k expert outputs. This scales total model capacity while keeping per-token inference FLOPs approximately constant."},
    {"prompt": "What is the mathematical formulation of LoRA?", "response": "Low-Rank Adaptation (LoRA) freezes pretrained weight matrix W_0 in R^(d x k) and parameterizes weight updates as Delta W = (alpha / r) * B * A, where B in R^(d x r) and A in R^(r x k) with rank r << min(d, k). Forward computation becomes h = W_0 x + (alpha / r) B A x. During inference, Delta W can be directly folded into W_0 without latency overhead."},

    # Mathematics & Philosophy
    {"prompt": "Explain the significance of Godel's Incompleteness Theorems.", "response": "Godel's First Incompleteness Theorem states that any consistent formal system capable of expressing elementary arithmetic contains true propositions that cannot be proven within the system. The Second Theorem proves that such a system cannot demonstrate its own consistency. Philosophically, this limits purely formalist programs and underscores the fundamental open-endedness of mathematical truth."},
    {"prompt": "What is the Hard Problem of Consciousness?", "response": "Formulated by David Chalmers, the Hard Problem contrasts 'easy problems' of cognitive science (processing sensory inputs, focusing attention, verbal reports) with the question of why and how physical brain processes give rise to subjective qualia—the inner qualitative experience of what it is 'like' to feel or perceive."}
]


def load_dataset_file(filepath: str) -> List[Dict[str, str]]:
    """
    Ingest samples from .jsonl, .json, .csv, or .md formats into standard [{"prompt": "...", "response": "..."}].
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file '{filepath}' does not exist.")

    ext = os.path.splitext(filepath)[1].lower()
    samples: List[Dict[str, str]] = []

    if ext == ".jsonl":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                p = item.get("prompt") or item.get("instruction") or item.get("input", "")
                r = item.get("response") or item.get("output", "")
                if p and r:
                    samples.append({"prompt": str(p).strip(), "response": str(r).strip()})

    elif ext == ".json":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    p = item.get("prompt") or item.get("instruction") or item.get("input", "")
                    r = item.get("response") or item.get("output", "")
                    if p and r:
                        samples.append({"prompt": str(p).strip(), "response": str(r).strip()})

    elif ext == ".csv":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("prompt") or row.get("instruction") or row.get("input", "")
                r = row.get("response") or row.get("output", "")
                if p and r:
                    samples.append({"prompt": str(p).strip(), "response": str(r).strip()})

    elif ext in [".md", ".txt"]:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        # Parse Q: ... A: ... or User: ... Assistant: ... blocks
        pairs = re.findall(r'(?:Q:|User:|Prompt:)\s*(.*?)\s*(?:A:|Assistant:|Response:)\s*(.*?)(?=(?:Q:|User:|Prompt:)|\Z)', content, re.DOTALL | re.IGNORECASE)
        for p, r in pairs:
            if p.strip() and r.strip():
                samples.append({"prompt": p.strip(), "response": r.strip()})

    return samples


def generate_preset_dataset(output_path: str = "data/lunaite_training_data.jsonl") -> str:
    """Generate and write the multi-domain training dataset."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in MULTI_DOMAIN_PRESETS:
            f.write(json.dumps(item) + "\n")
    return output_path
