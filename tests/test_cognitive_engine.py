import pytest
import torch
from lunaite.core.adaptive_moe import AdaptiveMoEConfig, AdaptiveEntropyRouter, AdaptiveMoELayer
from lunaite.agent.cognitive_engine import NeuroSymbolicMCTSEngine, FormalASTVerifier, ReasoningStepNode
from lunaite.core.speculative import SpeculativeDecoder, SpeculativeConfig


class TestAdaptiveMoE:
    def test_entropy_router_shapes_and_metrics(self):
        config = AdaptiveMoEConfig(
            num_experts=8,
            hidden_dim=256,
            moe_intermediate_dim=512,
            min_k=1,
            max_k=4
        )
        router = AdaptiveEntropyRouter(config)
        hidden = torch.randn(2, 16, 256)  # [batch=2, seq=16, dim=256]

        weights, indices, aux_loss, metrics = router(hidden)

        assert weights.shape == (32, 4)  # [32 tokens, max_k=4]
        assert indices.shape == (32, 4)
        assert aux_loss.item() >= 0.0
        assert "mean_entropy" in metrics
        assert "mean_dynamic_k" in metrics
        assert 1.0 <= metrics["mean_dynamic_k"] <= 4.0

    def test_adaptive_moe_layer_forward(self):
        config = AdaptiveMoEConfig(
            num_experts=4,
            hidden_dim=128,
            moe_intermediate_dim=256,
            min_k=1,
            max_k=2
        )
        layer = AdaptiveMoELayer(config)
        hidden = torch.randn(1, 8, 128)

        out, aux_loss, metrics = layer(hidden)
        assert out.shape == hidden.shape
        assert not torch.isnan(out).any()


class TestCognitiveEngine:
    def test_ast_verifier_valid_and_invalid_code(self):
        valid_code = "def solve(x):\n    return x * 2 + 1\n"
        invalid_code = "def broken(:\n return"

        is_valid, msg = FormalASTVerifier.verify_syntax(valid_code)
        assert is_valid is True

        is_invalid, err_msg = FormalASTVerifier.verify_syntax(invalid_code)
        assert is_invalid is False
        assert "SyntaxError" in err_msg

    def test_mcts_deliberation_trajectory(self):
        engine = NeuroSymbolicMCTSEngine(max_simulations=10, max_depth=3)
        res = engine.deliberate("Prove that primes greater than 2 are odd.")

        assert "optimal_trajectory" in res
        assert len(res["optimal_trajectory"]) > 0
        assert res["simulations_count"] == 10
        assert res["mean_prm_confidence"] > 0.0


class TestSpeculativeDecoding:
    def test_speculative_acceptance_simulation(self):
        decoder = SpeculativeDecoder(SpeculativeConfig(acceptance_threshold=0.8))
        draft = [0.9, 0.85, 0.82, 0.5]
        target = [0.95, 0.88, 0.83, 0.2]

        emitted, speedup = decoder.simulate_speculation_step(draft, target)
        assert emitted >= 1
        assert speedup > 0.0
