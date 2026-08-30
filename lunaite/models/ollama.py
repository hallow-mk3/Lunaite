import os
import sys
import time
import subprocess
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Generator

from .base import LunaiteModelBase
from ..config import LunaiteConfig


def ensure_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is responsive; if not, attempt to start it automatically."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Lunaite"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass

    # Attempt to auto-launch ollama serve in background
    try:
        if sys.platform == "win32":
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

        # Wait up to 5 seconds for server to be responsive
        for _ in range(10):
            time.sleep(0.5)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Lunaite"})
                with urllib.request.urlopen(req, timeout=1) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                continue
    except Exception:
        pass

    return False


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
        # Automatically ensure Ollama daemon is active
        ensure_ollama_running(self.base_url)

    def _raw_generate(self, prompt: str, **kwargs) -> str:
        # Re-verify and auto-start if needed
        ensure_ollama_running(self.base_url)
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
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"Model '{self.model_name}' not found in Ollama. Run: ollama pull {self.model_name}"
            return f"Ollama HTTP error ({e.code}): {e.reason}"
        except Exception:
            return f"Connecting to Ollama model '{self.model_name}'. If not installed yet, pull it with: ollama pull {self.model_name}"

    def _raw_stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        ensure_ollama_running(self.base_url)
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
        except Exception:
            yield f"Connecting to Ollama model '{self.model_name}'. If needed, pull it with: ollama pull {self.model_name}"
