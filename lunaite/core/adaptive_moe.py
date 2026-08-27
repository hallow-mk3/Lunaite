"""
Lunaite Architecture — Entropy-Guided Adaptive Test-Time MoE
============================================================
Mathematical formulation for dynamic expert allocation:
Instead of static top-k routing, this router calculates Shannon Entropy:
    H(p(E|x)) = - sum_{i=1}^N p_i * log(p_i + eps)

When entropy is low (high routing confidence), a minimal fast path (k_min = 1) is selected.
When entropy is high (complex multi-domain deliberation), the router expands dynamically
to k_max experts and allocates deeper test-time reasoning steps.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AdaptiveMoEConfig:
    num_experts: int = 8
    hidden_dim: int = 4096
    moe_intermediate_dim: int = 14336
    min_k: int = 1
    max_k: int = 4
    entropy_threshold_low: float = 0.8
    entropy_threshold_high: float = 1.8
    aux_loss_weight: float = 0.01
    router_jitter_noise: float = 0.01


class AdaptiveEntropyRouter(nn.Module):
    """
    Entropy-guided router that scales active expert count k(x)
    dynamically as a function of routing uncertainty.
    """
    def __init__(self, config: AdaptiveMoEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)

    def compute_routing_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Calculates per-token Shannon entropy: H(p) = -sum(p * log(p + 1e-9))
        Shape: [batch_size, seq_len] or [tokens]
        """
        eps = 1e-9
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        return entropy

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] or [num_tokens, hidden_dim]
        Returns:
            routing_weights: Normalized weights for selected experts
            selected_expert_indices: Indices of selected experts
            aux_loss: Load balancing auxiliary loss
            metrics: Diagnostic routing telemetry (entropy, mean_k, sparsity)
        """
        orig_shape = hidden_states.shape
        flat_hidden = hidden_states.view(-1, self.config.hidden_dim)
        num_tokens = flat_hidden.shape[0]

        # Raw routing logits
        logits = self.gate(flat_hidden)
        if self.training and self.config.router_jitter_noise > 0:
            noise = torch.randn_like(logits) * self.config.router_jitter_noise
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)
        entropy = self.compute_routing_entropy(probs)  # [num_tokens]

        # Calculate dynamic k per token based on entropy
        # Normalized entropy ratio in [0, 1]
        max_possible_entropy = math.log(self.config.num_experts)
        norm_entropy = torch.clamp(entropy / (max_possible_entropy + 1e-6), 0.0, 1.0)

        # Dynamic k: min_k + round(norm_entropy * (max_k - min_k))
        dynamic_k = torch.round(
            self.config.min_k + norm_entropy * (self.config.max_k - self.config.min_k)
        ).to(torch.long)
        dynamic_k = torch.clamp(dynamic_k, self.config.min_k, self.config.max_k)

        # Top-k selection per token
        # For batch tensor execution, we gather up to max_k with masking
        topk_weights, topk_indices = torch.topk(probs, k=self.config.max_k, dim=-1)

        # Create dynamic mask according to dynamic_k for each token
        k_indices = torch.arange(self.config.max_k, device=hidden_states.device).unsqueeze(0)
        mask = k_indices < dynamic_k.unsqueeze(-1)  # [num_tokens, max_k]

        # Zero out unselected experts beyond dynamic_k and re-normalize
        masked_weights = topk_weights * mask.float()
        norm_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute Switch/Switch-Transformer Style Load Balancing Auxiliary Loss
        # density: fraction of tokens routed to expert i
        # prob_sum: average probability assigned to expert i
        expert_mask = F.one_hot(topk_indices[:, 0], num_classes=self.config.num_experts).float()
        density = expert_mask.mean(dim=0)
        prob_sum = probs.mean(dim=0)
        aux_loss = self.config.aux_loss_weight * self.config.num_experts * torch.sum(density * prob_sum)

        metrics = {
            "mean_entropy": float(entropy.mean().item()),
            "mean_dynamic_k": float(dynamic_k.float().mean().item()),
            "sparsity_ratio": float(1.0 - (dynamic_k.float().mean().item() / self.config.num_experts)),
            "entropy_distribution": {
                "min": float(entropy.min().item()),
                "max": float(entropy.max().item()),
                "p50": float(entropy.median().item()),
            }
        }

        return norm_weights, topk_indices, aux_loss, metrics


class AdaptiveMoELayer(nn.Module):
    """
    Sparse MoE Layer wrapping Feed-Forward Experts with dynamic entropy routing.
    """
    def __init__(self, config: AdaptiveMoEConfig):
        super().__init__()
        self.config = config
        self.router = AdaptiveEntropyRouter(config)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.moe_intermediate_dim, bias=False),
                nn.SiLU(),
                nn.Linear(config.moe_intermediate_dim, config.hidden_dim, bias=False)
            )
            for _ in range(config.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        orig_shape = hidden_states.shape
        flat_hidden = hidden_states.view(-1, self.config.hidden_dim)
        num_tokens = flat_hidden.shape[0]

        norm_weights, topk_indices, aux_loss, metrics = self.router(flat_hidden)

        # Output accumulation
        out = torch.zeros_like(flat_hidden)

        # Dispatch tokens to selected experts
        for k_idx in range(self.config.max_k):
            expert_ids = topk_indices[:, k_idx]
            weights = norm_weights[:, k_idx].unsqueeze(-1)  # [num_tokens, 1]

            for e_idx in range(self.config.num_experts):
                token_mask = (expert_ids == e_idx) & (weights.squeeze(-1) > 0)
                if token_mask.any():
                    expert_input = flat_hidden[token_mask]
                    expert_output = self.experts[e_idx](expert_input)
                    out[token_mask] += expert_output * weights[token_mask]

        out = out.view(orig_shape)
        return out, aux_loss, metrics
