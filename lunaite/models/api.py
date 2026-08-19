"""
Lunaite Architecture — Universal Cloud API Model Adapter
========================================================
Wraps any standard OpenAI-compatible API or cloud endpoint (OpenAI, Anthropic,
Google Gemini, Groq, DeepSeek, vLLM, LocalAI) with Lunaite Architecture.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Generator

from .base import LunaiteModelBase
from ..config import LunaiteConfig


class LunaiteAPIModel(LunaiteModelBase):
    """
    Wraps any OpenAI-compatible API endpoint with Lunaite Architecture.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        config: Optional[LunaiteConfig] = None
    ):
        super().__init__(config)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def _raw_generate(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", self.config.cognitive.temperature),
            "top_p": kwargs.get("top_p", self.config.cognitive.top_p),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.URLError as e:
            return f"[Lunaite API Error]: {e}"

    def _raw_stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        res = self._raw_generate(prompt, **kwargs)
        yield res
