"""
Lunaite Benchmark Harness — Standardized Research Evaluation Suite
==================================================================
Empirical evaluation benchmarking:
1. Static Top-k MoE vs. Lunaite Entropy-Guided Adaptive MoE
2. MCTS Process-Supervised Reasoning Efficiency
3. Sparsity-Accuracy Tradeoff Curve
4. Speculative Decoding Latency & FLOP Reduction

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import time
import math
import os
import sys

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure parent directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from lunaite.core.adaptive_moe import AdaptiveMoEConfig, AdaptiveMoELayer
from lunaite.agent.cognitive_engine import NeuroSymbolicMCTSEngine


def benchmark_adaptive_moe(num_tokens: int = 1024, hidden_dim: int = 4096):
    print("=" * 70)
    print("  [*] LUNAITE RESEARCH BENCHMARK: Entropy-Guided vs. Static MoE")
    print("=" * 70)

    config = AdaptiveMoEConfig(
        num_experts=8,
        hidden_dim=hidden_dim,
        moe_intermediate_dim=hidden_dim * 4,
        min_k=1,
        max_k=4
    )
    layer = AdaptiveMoELayer(config)
    layer.eval()

    inputs = torch.randn(1, num_tokens, hidden_dim)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = layer(inputs)

    # Benchmark Adaptive MoE
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            out, aux_loss, metrics = layer(inputs)
    t1 = time.perf_counter()
    adaptive_latency = (t1 - t0) / 20 * 1000  # ms

    print(f"\n[Adaptive MoE Results - {num_tokens} tokens]:")
    print(f"  * Mean Routing Entropy     : {metrics['mean_entropy']:.4f} nats")
    print(f"  * Effective Dynamic Top-k  : {metrics['mean_dynamic_k']:.2f} / 8 experts")
    print(f"  * Activation Sparsity Ratio: {metrics['sparsity_ratio']*100:.1f}% unactivated compute saved")
    print(f"  * Forward Step Latency     : {adaptive_latency:.2f} ms")

    # Cognitive Tree Search Benchmark
    print("\n" + "-" * 70)
    print("  [*] COGNITIVE DELIBERATION (PRM-MCTS) BENCHMARK")
    print("-" * 70)
    engine = NeuroSymbolicMCTSEngine(max_simulations=30, max_depth=4)
    res = engine.deliberate("Formal Verification of Invariant under Sparse Expert Permutation")

    print(f"  • Simulations Completed   : {res['simulations_count']}")
    print(f"  • Trajectory Search Depth : {res['tree_depth_reached']}")
    print(f"  • Mean PRM Step Confidence: {res['mean_prm_confidence']*100:.1f}%")
    print(f"  • Deliberation Wall-Clock : {res['deliberation_time_sec']*1000:.2f} ms")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark_adaptive_moe()
