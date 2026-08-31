import os
import sys
import time
import subprocess
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator

from .base import LunaiteModelBase
from ..config import LunaiteConfig


def _get_hidden_process_kwargs() -> Dict[str, Any]:
    """Return subprocess kwargs to run background processes completely hidden without flashing console windows."""
    kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    return kwargs


def ensure_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is responsive; if not, attempt to start it silently in background."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Lunaite"})
        with urllib.request.urlopen(req, timeout=1) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass

    # Attempt to auto-launch ollama serve completely silently in background
    try:
        # Check standard Windows paths if 'ollama' is not on standard PATH
        ollama_bin = "ollama"
        if sys.platform == "win32":
            local_app = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
            if local_app.is_file():
                ollama_bin = str(local_app)

        hidden_kwargs = _get_hidden_process_kwargs()
        if sys.platform == "win32":
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                **hidden_kwargs
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True
            )

        # Wait up to 3 seconds for server to respond
        for _ in range(6):
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
        model_name: str = "lunaite-ai",
        base_url: str = "http://localhost:11434",
        config: Optional[LunaiteConfig] = None
    ):
        super().__init__(config=config)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        ensure_ollama_running(self.base_url)

    def _raw_generate(self, prompt: str, **kwargs) -> str:
        """Call Ollama /api/generate endpoint directly."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.cognitive.temperature),
                "num_predict": kwargs.get("max_tokens", 2048)
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "")
        except urllib.error.URLError as e:
            return f"[Lunaite Connection Error]: Unable to reach Ollama at {self.base_url} ({e})"

    def _raw_stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream tokens directly from Ollama /api/generate."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.cognitive.temperature),
                "num_predict": kwargs.get("max_tokens", 2048)
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
        except urllib.error.URLError as e:
            yield f"[Lunaite Stream Error]: Unable to stream from Ollama ({e})"
