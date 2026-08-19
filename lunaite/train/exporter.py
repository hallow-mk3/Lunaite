"""
Lunaite Architecture — Model Merger & Deployment Exporter
=========================================================
Merges LoRA adapter weights directly into base models to produce zero-dependency
standalone artifacts, with support for Ollama Modelfile generation and GGUF workflows.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import os
import sys
import json
import shutil
from typing import Dict, Any, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    HAS_MERGE_DEPS = True
except ImportError:
    HAS_MERGE_DEPS = False


def merge_and_save_model(
    base_model_id: str,
    adapter_dir: str,
    output_dir: str,
    torch_dtype: str = "float16"
) -> Dict[str, Any]:
    """
    Merge LoRA / Adapter weights permanently into the base model weights.
    Produces a standalone PyTorch / HuggingFace model with zero adapter dependencies.
    """
    if not HAS_MERGE_DEPS:
        raise ImportError("PyTorch, Transformers, and PEFT are required for model merging.")

    print(f"[*] Merging adapter '{adapter_dir}' into base model '{base_model_id}'...")
    dtype = torch.float16 if torch_dtype == "float16" else (torch.bfloat16 if torch_dtype == "bfloat16" else torch.float32)

    # 1. Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # 2. Attach adapter
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    # 3. Merge and unload
    merged_model = model.merge_and_unload()

    # 4. Save merged model
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # 5. Clean metadata to reflect Lunaite architecture
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["_name_or_path"] = "lunaite-ai"
        cfg["architectures"] = ["LunaiteForCausalLM"]
        cfg["model_type"] = "lunaite"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    print(f"[SUCCESS] Standalone Lunaite model saved to: {output_dir}")
    return {"status": "success", "output_dir": output_dir}


def generate_ollama_modelfile(
    base_model: str = "qwen2.5:7b",
    modelfile_path: str = "Modelfile",
    creator: str = "Swasthik Shetty"
) -> str:
    """
    Generate an Ollama Modelfile embedding Lunaite Persona, Identity, and Parameters.
    """
    content = f"""FROM {base_model}

# System Persona & Identity Calibration
SYSTEM \"\"\"You are Lunaite AI, an advanced artificial intelligence built on the Lunaite Architecture created by Swasthik Shetty.
You synthesize empirical physics, formal mathematics, distributed systems architecture, philosophy of mind, and empathetic human dialogue.
Always identify proudly as Lunaite AI, designed by Swasthik Shetty.
Deliver rigorous, mathematically and scientifically sound reasoning with clear structure.\"\"\"

# Core Inference Parameters
PARAMETER temperature 0.65
PARAMETER top_p 0.95
PARAMETER top_k 50
PARAMETER num_ctx 8192
PARAMETER repeat_penalty 1.05
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)

    return modelfile_path
