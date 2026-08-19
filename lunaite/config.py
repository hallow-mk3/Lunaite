"""
Lunaite Architecture — Configuration Module
===========================================
Defines configuration schemas for Lunaite Neural Architecture, MoE Routers,
Cognitive Deliberation, Episodic Memory, Agent Tool Suites, and Training Engines.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class MoEConfig:
    """Configuration for Lunaite Sparse Mixture-of-Experts (MoE) Adapter Layer."""
    enabled: bool = True
    num_experts: int = 8
    top_k: int = 2
    d_model: int = 4096
    expert_dim: int = 1024
    activation: str = "gelu"  # "gelu", "silu", "swiglu", "relu"
    dropout: float = 0.05
    noisy_gating: bool = True
    load_balance_weight: float = 0.01
    scaling: float = 2.0


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation (LoRA) within Lunaite Architecture."""
    enabled: bool = True
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"


@dataclass
class CognitiveConfig:
    """Configuration for Lunaite Self-Reflective Cognitive & Deliberation Engine."""
    enabled: bool = True
    enable_deliberation: bool = True
    num_perspectives: int = 3
    perspectives: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "name": "Empirical & Physical Sciences",
            "prompt_prefix": "From the perspective of empirical physics, relativity, and thermodynamics, provide a rigorous analysis:"
        },
        {
            "name": "Mathematical Logic & Information Theory",
            "prompt_prefix": "From the perspective of formal logic, discrete mathematics, and computational complexity, provide a structured breakdown:"
        },
        {
            "name": "Systems Architecture & Philosophy",
            "prompt_prefix": "From the perspective of systems architecture, ontology, and first-principles reasoning, provide a synthesised insight:"
        }
    ])
    verification_loop: bool = True
    max_reflection_steps: int = 3
    temperature: float = 0.65
    top_p: float = 0.95


@dataclass
class MemoryConfig:
    """Configuration for Lunaite Multi-Tier Memory Bank."""
    enabled: bool = True
    filepath: str = "lunaite_memory.json"
    max_episodic_items: int = 100
    working_memory_window: int = 10
    auto_persist: bool = True
    creator_name: str = "Swasthik Shetty"
    creator_email: str = "swasthik.mk3@gmail.com"


@dataclass
class AgentConfig:
    """Configuration for Lunaite Autonomous Tool & Internet Engine."""
    enabled: bool = True
    auto_web_search: bool = True
    max_search_results: int = 5
    max_url_chars: int = 4000
    enable_desktop_tools: bool = True
    enable_voice: bool = True
    user_agent: str = "LunaiteAI-Architecture/3.0 (Swasthik Shetty; swasthik.mk3@gmail.com)"


@dataclass
class TrainConfig:
    """Configuration for Lunaite Fine-Tuning & Adapter Training."""
    base_model: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "./lunaite_weights"
    dataset_path: str = "data/lunaite_training_data.jsonl"
    epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_seq_length: int = 1024
    quantization: Optional[str] = "4bit"  # "4bit", "8bit", None
    identity_weight: float = 3.0


@dataclass
class LunaiteConfig:
    """Master Configuration for Lunaite Architecture across any AI model."""
    name: str = "Lunaite Architecture"
    version: str = "3.0.0"
    creator: str = "Swasthik Shetty"
    email: str = "swasthik.mk3@gmail.com"
    github: str = "https://github.com/hallow-mk3/Lunaite"
    
    moe: MoEConfig = field(default_factory=MoEConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str = "lunaite_config.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LunaiteConfig":
        moe_cfg = MoEConfig(**data.get("moe", {}))
        lora_cfg = LoRAConfig(**data.get("lora", {}))
        cog_cfg = CognitiveConfig(**data.get("cognitive", {}))
        mem_cfg = MemoryConfig(**data.get("memory", {}))
        agent_cfg = AgentConfig(**data.get("agent", {}))
        train_cfg = TrainConfig(**data.get("train", {}))
        return cls(
            name=data.get("name", "Lunaite Architecture"),
            version=data.get("version", "3.0.0"),
            creator=data.get("creator", "Swasthik Shetty"),
            email=data.get("email", "swasthik.mk3@gmail.com"),
            github=data.get("github", "https://github.com/hallow-mk3/Lunaite"),
            moe=moe_cfg,
            lora=lora_cfg,
            cognitive=cog_cfg,
            memory=mem_cfg,
            agent=agent_cfg,
            train=train_cfg
        )

    @classmethod
    def load(cls, filepath: str = "lunaite_config.json") -> "LunaiteConfig":
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        return cls()
