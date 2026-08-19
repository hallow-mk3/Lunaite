"""
Lunaite Architecture — Base Model Abstraction
=============================================
Defines the base interface for any model wrapped by Lunaite Architecture.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, Callable

from ..config import LunaiteConfig
from ..core.cognitive import LunaiteCognitiveEngine
from ..core.memory import LunaiteMemory
from ..agent.suite import LunaiteAgent


class LunaiteModelBase(ABC):
    """
    Abstract Base Class for models empowered by Lunaite Architecture.
    """
    def __init__(self, config: Optional[LunaiteConfig] = None):
        self.config = config or LunaiteConfig()
        self.memory = LunaiteMemory(self.config.memory) if self.config.memory.enabled else None
        self.cognitive = LunaiteCognitiveEngine(self.config.cognitive) if self.config.cognitive.enabled else None
        self.agent = LunaiteAgent(self.config.agent) if self.config.agent.enabled else None
        self.history: List[Dict[str, str]] = []

    @abstractmethod
    def _raw_generate(self, prompt: str, **kwargs) -> str:
        """Raw model generation without cognitive/agent wrapping."""
        pass

    @abstractmethod
    def _raw_stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Raw model streaming token generator."""
        pass

    def generate(self, prompt: str, use_deliberation: bool = False, use_agent: bool = True, **kwargs) -> str:
        """
        Generate response through Lunaite Cognitive and Agent Architecture.
        """
        # Inject memory context if available
        memory_ctx = self.memory.get_context_summary() if self.memory else ""
        system_prompt = self.cognitive.get_system_prompt(memory_ctx) if self.cognitive else ""

        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nLunaite AI:" if system_prompt else prompt

        def base_gen(p: str) -> str:
            return self._raw_generate(p, **kwargs)

        if use_deliberation and self.cognitive:
            response = self.cognitive.deliberate(prompt, base_gen)
        elif use_agent and self.agent:
            response = self.agent.process_prompt(prompt, base_gen)
        else:
            response = base_gen(full_prompt)

        # Store into memory
        if self.memory and len(response.split()) > 10:
            self.memory.add_insight(f"Q: {prompt[:60]}... -> A: {response[:80]}...")

        return response

    def chat(self, user_message: str, **kwargs) -> str:
        """Multi-turn conversation with stateful memory."""
        self.history.append({"role": "user", "content": user_message})
        response = self.generate(user_message, **kwargs)
        self.history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self):
        """Reset conversation history."""
        self.history.clear()
