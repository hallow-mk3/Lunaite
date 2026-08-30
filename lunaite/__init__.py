"""
Lunaite — Universal Modular AI Architecture Framework
=====================================================
A unified, high-performance architecture framework designed by Swasthik Shetty.
Attaches Sparse Mixture-of-Experts (MoE) residual routing, multi-perspective cognitive
deliberation, multi-tier episodic memory, and autonomous agent tool capabilities to ANY AI model.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
GitHub: https://github.com/hallow-mk3/Lunaite
License: MIT
"""

__version__ = "3.0.0"
__author__ = "Swasthik Shetty"
__email__ = "swasthik.mk3@gmail.com"

from .config import LunaiteConfig, MoEConfig, LoRAConfig, CognitiveConfig, MemoryConfig, AgentConfig
from .core.architecture import (
    LunaiteMoERouter,
    LunaiteExpert,
    LunaiteMoELayer,
    LunaiteArchitecturalAdapter,
    calculate_architecture_parameters,
)
from .core.cognitive import LunaiteCognitiveEngine, SYSTEM_PERSONA_PROMPT
from .core.memory import LunaiteMemory
from .agent.suite import LunaiteAgent
from .agent.tools import web_search, fetch_url, wiki_lookup, fetch_weather
from .agent.desktop import get_system_telemetry, take_screenshot, read_clipboard, write_clipboard, run_powershell
from .models.base import LunaiteModelBase
from .models.ollama import LunaiteOllamaModel
from .models.huggingface import LunaiteHuggingFaceModel
from .models.api import LunaiteAPIModel
from .models.wrapper import wrap, from_ollama, from_huggingface, from_api

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Config
    "LunaiteConfig",
    "MoEConfig",
    "LoRAConfig",
    "CognitiveConfig",
    "MemoryConfig",
    "AgentConfig",
    # Core Architecture
    "LunaiteMoERouter",
    "LunaiteExpert",
    "LunaiteMoELayer",
    "LunaiteArchitecturalAdapter",
    "calculate_architecture_parameters",
    "LunaiteCognitiveEngine",
    "SYSTEM_PERSONA_PROMPT",
    "LunaiteMemory",
    # Agent & Tools
    "LunaiteAgent",
    "web_search",
    "fetch_url",
    "wiki_lookup",
    "fetch_weather",
    "get_system_telemetry",
    "take_screenshot",
    "read_clipboard",
    "write_clipboard",
    "run_powershell",
    # Models & Universal Wrapper
    "LunaiteModelBase",
    "LunaiteOllamaModel",
    "LunaiteHuggingFaceModel",
    "LunaiteAPIModel",
    "wrap",
    "from_ollama",
    "from_huggingface",
    "from_api",
]
