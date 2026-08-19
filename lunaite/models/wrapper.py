"""
Lunaite Architecture — Universal Model Wrapper Factory
======================================================
Provides seamless one-line initialization to empower ANY AI model with Lunaite Architecture.

Examples:
    import lunaite

    # 1. Any Ollama Model
    model = lunaite.wrap("qwen2.5:7b")
    # or lunaite.from_ollama("llama3.1:8b")

    # 2. Any HuggingFace Model
    model = lunaite.wrap("meta-llama/Llama-3-8B-Instruct", backend="huggingface")

    # 3. Any API Model
    model = lunaite.wrap("gpt-4o", backend="api")

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

from typing import Union, Any, Optional
from .base import LunaiteModelBase
from .ollama import LunaiteOllamaModel
from .huggingface import LunaiteHuggingFaceModel
from .api import LunaiteAPIModel
from ..config import LunaiteConfig


def from_ollama(
    model_name: str = "lunaite-ai",
    base_url: str = "http://localhost:11434",
    config: Optional[LunaiteConfig] = None
) -> LunaiteOllamaModel:
    """Wrap any local Ollama model with Lunaite Architecture."""
    return LunaiteOllamaModel(model_name=model_name, base_url=base_url, config=config)


def from_huggingface(
    model_name_or_path: str,
    device: str = "auto",
    torch_dtype: str = "auto",
    config: Optional[LunaiteConfig] = None
) -> LunaiteHuggingFaceModel:
    """Wrap any Hugging Face model with Lunaite MoE and LoRA Architecture."""
    return LunaiteHuggingFaceModel(
        model_or_path=model_name_or_path,
        device=device,
        torch_dtype=torch_dtype,
        config=config
    )


def from_api(
    model_name: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    config: Optional[LunaiteConfig] = None
) -> LunaiteAPIModel:
    """Wrap any OpenAI-compatible API endpoint with Lunaite Architecture."""
    return LunaiteAPIModel(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        config=config
    )


def wrap(
    model: Union[str, Any],
    backend: Optional[str] = None,
    config: Optional[LunaiteConfig] = None,
    **kwargs
) -> LunaiteModelBase:
    """
    Universal factory to attach Lunaite Architecture to ANY model.
    
    Args:
        model: Model name/path string or pre-instantiated model object.
        backend: "ollama", "huggingface", "api", or None (auto-detect).
        config: Custom LunaiteConfig instance.
        **kwargs: Backend-specific arguments.
    """
    if isinstance(model, LunaiteModelBase):
        return model

    if backend == "huggingface" or "/" in str(model) and backend != "api":
        return from_huggingface(model, config=config, **kwargs)
    elif backend == "api":
        return from_api(model, config=config, **kwargs)
    elif backend == "ollama" or ":" in str(model) or backend is None:
        return from_ollama(str(model), config=config, **kwargs)
    else:
        return from_ollama(str(model), config=config, **kwargs)
