"""
Lunaite Architecture — Universal Ollama Model Adapter
=====================================================
Wraps ANY Ollama model (e.g. LLaMA 3, Qwen 2.5, Mistral, Gemma, Phi, DeepSeek)
with Lunaite Neural & Cognitive Architecture.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Generator

from .base import LunaiteModelBase
from ..config import LunaiteConfig


class LunaiteOllamaModel(LunaiteModelBase):
    """
    Empowers any local Ollama model with Lunaite Cognitive, Memory, and Agent Architecture.
    """
    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        config: Optional[LunaiteConfig] = None
    ):
        super().__init__(config)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def _raw_generate(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.cognitive.temperature),
                "top_p": kwargs.get("top_p", self.config.cognitive.top_p),
            }
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except urllib.error.URLError as e:
            return f"[Lunaite Ollama Error]: Could not reach Ollama at {self.base_url} ({e}). Please run 'ollama serve'."

    def _raw_stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.cognitive.temperature),
                "top_p": kwargs.get("top_p", self.config.cognitive.top_p),
            }
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        yield chunk.get("response", "")
                        if chunk.get("done", False):
                            break
                    except Exception:
                        continue
        except urllib.error.URLError as e:
            yield f"[Lunaite Ollama Connection Error]: {e}"
