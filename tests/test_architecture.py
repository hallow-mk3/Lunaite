"""
Unit tests for Lunaite Neural Architecture & Sparse MoE.
"""

import unittest
import torch
from lunaite.core.architecture import (
    LunaiteMoERouter,
    LunaiteExpert,
    LunaiteMoELayer,
    LunaiteArchitecturalAdapter,
    calculate_architecture_parameters
)


class TestLunaiteArchitecture(unittest.TestCase):

    def test_moe_router(self):
        d_model = 64
        num_experts = 4
        top_k = 2
        router = LunaiteMoERouter(d_model=d_model, num_experts=num_experts, top_k=top_k, noisy_gating=False)

        x = torch.randn(2, 5, d_model)
        weights, indices, aux_loss = router(x)

        self.assertEqual(weights.shape, (10, top_k))
        self.assertEqual(indices.shape, (10, top_k))
        self.assertTrue(torch.all(weights >= 0.0))
        self.assertTrue(torch.all(weights <= 1.0))
        self.assertTrue(aux_loss.item() >= 0.0)

    def test_moe_layer(self):
        d_model = 64
        layer = LunaiteMoELayer(d_model=d_model, num_experts=4, top_k=2, expert_dim=128)
        x = torch.randn(2, 4, d_model)
        out, aux_loss = layer(x)

        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_architectural_adapter(self):
        d_model = 64
        adapter = LunaiteArchitecturalAdapter(d_model=d_model, rank=16, alpha=32, num_experts=2)
        x = torch.randn(2, 4, d_model)
        out, aux_loss = adapter(x)

        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_parameter_calculation(self):
        params = calculate_architecture_parameters(
            base_params=7_000_000_000,
            d_model=4096,
            num_layers=32,
            num_experts=8,
            expert_dim=1024,
            lora_rank=64
        )
        self.assertEqual(params["base_parameters"], 7_000_000_000)
        self.assertTrue(params["total_added_parameters"] > 0)
        self.assertTrue(params["active_parameters_per_token"] < params["total_model_parameters"])


if __name__ == "__main__":
    unittest.main()
