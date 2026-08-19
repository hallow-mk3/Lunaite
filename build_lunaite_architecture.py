"""
Lunaite Architecture — Neural Adapter & Parameter Calculator
============================================================
Demonstrates the instantiation of Lunaite Architectural Adapter and calculates
exact parameter counts, active expert footprints, and sparsity metrics.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
"""

import json
import torch
from lunaite.core.architecture import LunaiteArchitecturalAdapter, calculate_architecture_parameters

print("=== Lunaite Architecture Neural Adapter & Parameter Engine ===")

# 1. Instantiate Adapter
adapter = LunaiteArchitecturalAdapter(d_model=4096, rank=64, alpha=128, num_experts=8)
trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)

print(f"Total Added Architectural Adapter Parameters: {trainable_params:,}")
print("Target Modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'moe_experts']")
print("Domains Integrated: Astrophysics, AI/ML, Mathematics, Software Engineering, Philosophy, Dialogue")

# 2. Calculate Rigorous Parameters across 32-layer 7B foundation
stats = calculate_architecture_parameters(
    base_params=7_000_000_000,
    d_model=4096,
    num_layers=32,
    num_experts=8,
    expert_dim=1024,
    lora_rank=64
)

print("\n--- Architecture Parameter Breakdown ---")
print(f"Base Model:          {stats['base_parameters_formatted']}")
print(f"Added LoRA/MoE:      +{stats['total_added_parameters_formatted']}")
print(f"Total Capacity:      {stats['total_model_parameters_formatted']}")
print(f"Active per Token:    {stats['active_parameters_formatted']}")
print(f"Sparsity:            {stats['sparsity_ratio']}")

with open("lunaite_architecture_config.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("\nSaved architecture configuration to 'lunaite_architecture_config.json'.")
