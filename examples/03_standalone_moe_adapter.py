"""
Example 3: Standalone PyTorch Lunaite Sparse MoE Adapter
========================================================
Demonstrates using the core LunaiteMoELayer and LunaiteArchitecturalAdapter
directly in PyTorch models with load-balancing auxiliary loss computation.
"""

import torch
import lunaite

# 1. Instantiate Lunaite Sparse MoE Layer
d_model = 4096
num_experts = 8
top_k = 2

moe_layer = lunaite.LunaiteMoELayer(
    d_model=d_model,
    num_experts=num_experts,
    top_k=top_k,
    expert_dim=1024,
    activation="gelu"
)

# 2. Forward pass with batch of token embeddings
batch_size = 2
seq_len = 16
x = torch.randn(batch_size, seq_len, d_model)

output, aux_loss = moe_layer(x)

print(f"Input shape:     {x.shape}")
print(f"Output shape:    {output.shape}")
print(f"MoE Aux Loss:    {aux_loss.item():.6f}")

# 3. Compute parameter statistics
stats = lunaite.calculate_architecture_parameters(
    base_params=7_000_000_000,
    d_model=4096,
    num_layers=32,
    num_experts=8,
    expert_dim=1024
)
print(f"\nArchitecture Summary:")
print(f"- Base Model Scale:      {stats['base_parameters_formatted']}")
print(f"- Added MoE Parameters:  {stats['total_added_parameters_formatted']}")
print(f"- Total Parameters:      {stats['total_model_parameters_formatted']}")
print(f"- Active per Token:      {stats['active_parameters_formatted']}")
print(f"- Sparsity Ratio:        {stats['sparsity_ratio']}")
