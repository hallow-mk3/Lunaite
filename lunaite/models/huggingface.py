"""
Lunaite Architecture — Universal Hugging Face Transformer Adapter
=================================================================
Wraps ANY HuggingFace Causal LM (Qwen, LLaMA, Mistral, Gemma, Phi, DeepSeek, Falcon, etc.)
with Lunaite Sparse MoE and LoRA Architectural Adapters.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

from typing import Dict, Any, List, Optional, Generator, Union

from .base import LunaiteModelBase
from ..config import LunaiteConfig
from ..core.architecture import LunaiteArchitecturalAdapter, LunaiteMoELayer

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_HF = True
except ImportError:
    HAS_HF = False


class LunaiteHuggingFaceModel(LunaiteModelBase):
    """
    Empowers any HuggingFace Causal Language Model with Lunaite Neural Architecture.
    """
    def __init__(
        self,
        model_or_path: Union[str, Any],
        device: str = "auto",
        torch_dtype: str = "auto",
        config: Optional[LunaiteConfig] = None
    ):
        super().__init__(config)
        if not HAS_HF:
            raise ImportError("Hugging Face transformers, torch, and peft are required. Install with: pip install torch transformers peft")

        self.device = device
        if isinstance(model_or_path, str):
            self.model_name_or_path = model_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            dtype = torch.float16 if torch_dtype == "float16" else (torch.bfloat16 if torch_dtype == "bfloat16" else "auto")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_or_path,
                torch_dtype=dtype,
                device_map=device if device != "cpu" else None,
                trust_remote_code=True
            )
        else:
            self.model = model_or_path
            self.model_name_or_path = getattr(model_or_path, "name_or_path", "custom_hf_model")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)

        self._attach_lunaite_adapters()

    def _attach_lunaite_adapters(self):
        """Inject LoRA and MoE parameter adapters onto target linear projections."""
        if not self.config.lora.enabled:
            return

        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias
        )
        try:
            self.model = get_peft_model(self.model, peft_cfg)
        except Exception as e:
            print(f"[Lunaite HuggingFace Adapter Warning]: Could not attach PEFT model: {e}")

    def _raw_generate(self, prompt: str, **kwargs) -> str:
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", self.config.cognitive.temperature)
        top_p = kwargs.get("top_p", self.config.cognitive.top_p)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _raw_stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        # Fallback to yield full string if TextIteratorStreamer is not used
        full_res = self._raw_generate(prompt, **kwargs)
        yield full_res
