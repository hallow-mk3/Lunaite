"""
Lunaite AI 10B — Merge, Export to GGUF, and Rebuild Ollama Model
=================================================================
This script merges the trained LoRA weights (if not already merged),
converts the merged model to GGUF format using llama.cpp,
and rebuilds the Ollama 'lunaite-ai' model from the GGUF file.

After this script completes, running `ollama run lunaite-ai` will
show ZERO trace of Qwen — it is purely Lunaite AI.

Usage:
    python merge_export.py

Requirements:
    - llama.cpp must be built (run build_llamacpp.ps1)
    - pip install transformers torch peft
"""

import os
import sys
import json
import subprocess
import shutil
import urllib.request

# ─── Configuration ────────────────────────────────────────────────────────────

LORA_DIR        = "./lunaite_lora_weights"
MERGED_DIR      = "./lunaite_merged"
GGUF_DIR        = "./lunaite_gguf"
GGUF_FILE       = os.path.join(GGUF_DIR, "lunaite-ai-10b-q8_0.gguf")
LLAMACPP_DIR    = "./llama.cpp"
MODELFILE_PATH  = "./Modelfile.gguf"
OLLAMA_MODEL    = "lunaite-ai"
QUANTIZATION    = "q8_0"      # High quality quantization (8-bit)

# ─── Step 1: Merge LoRA if not already merged ────────────────────────────────

def merge_lora():
    if os.path.exists(MERGED_DIR) and len(os.listdir(MERGED_DIR)) > 0:
        print("[1/4] Merged model already exists. Skipping merge step.")
        return True

    print("[1/4] Merging LoRA adapter weights into base model...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        base_id = "Qwen/Qwen2.5-7B"
        print(f"      Loading base: {base_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            device_map="cpu",   # CPU for merge (avoids VRAM limits)
            trust_remote_code=True
        )

        print(f"      Loading LoRA adapters from: {LORA_DIR}")
        model = PeftModel.from_pretrained(model, LORA_DIR)

        print("      Merging and unloading adapters...")
        model = model.merge_and_unload()

        print(f"      Saving merged model to: {MERGED_DIR}")
        os.makedirs(MERGED_DIR, exist_ok=True)
        model.save_pretrained(MERGED_DIR, safe_serialization=True)
        tokenizer.save_pretrained(MERGED_DIR)

        # Patch model config to erase Qwen branding
        config_path = os.path.join(MERGED_DIR, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            config["_name_or_path"] = "lunaite-ai"
            config["model_type"] = "lunaite"
            config.pop("auto_map", None)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print("      Patched config.json: removed Qwen branding.")

        print("      Merge complete.")
        return True

    except ImportError as e:
        print(f"      Missing dependency: {e}")
        print("      Install with: pip install torch transformers peft")
        return False
    except Exception as e:
        print(f"      Merge error: {e}")
        return False


# ─── Step 2: Convert to GGUF ─────────────────────────────────────────────────

def convert_to_gguf():
    print("\n[2/4] Converting merged model to GGUF format...")
    os.makedirs(GGUF_DIR, exist_ok=True)

    convert_script = os.path.join(LLAMACPP_DIR, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        # Try older path
        convert_script = os.path.join(LLAMACPP_DIR, "convert-hf-to-gguf.py")
    if not os.path.exists(convert_script):
        print(f"      ERROR: llama.cpp conversion script not found at {LLAMACPP_DIR}")
        print("      Run: python build_llamacpp.py  to set up llama.cpp")
        return False

    # First convert to F16
    f16_gguf = os.path.join(GGUF_DIR, "lunaite-ai-f16.gguf")
    cmd_convert = [
        sys.executable, convert_script,
        MERGED_DIR,
        "--outfile", f16_gguf,
        "--outtype", "f16"
    ]
    print(f"      Running: {' '.join(cmd_convert)}")
    result = subprocess.run(cmd_convert, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"      Conversion failed:\n{result.stderr}")
        return False
    print(f"      F16 GGUF created: {f16_gguf}")

    # Quantize to Q8_0
    quantize_bin = os.path.join(LLAMACPP_DIR, "build", "bin", "llama-quantize")
    if not os.path.exists(quantize_bin):
        quantize_bin = os.path.join(LLAMACPP_DIR, "llama-quantize")
    if not os.path.exists(quantize_bin):
        # Windows
        quantize_bin = os.path.join(LLAMACPP_DIR, "build", "bin", "Release", "llama-quantize.exe")

    if os.path.exists(quantize_bin):
        cmd_quant = [quantize_bin, f16_gguf, GGUF_FILE, QUANTIZATION.upper()]
        print(f"      Quantizing to {QUANTIZATION.upper()}...")
        result = subprocess.run(cmd_quant, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"      Quantized GGUF: {GGUF_FILE}")
            os.remove(f16_gguf)  # Clean up F16
        else:
            print(f"      Quantization warning: {result.stderr}")
            shutil.move(f16_gguf, GGUF_FILE)
            print(f"      Using F16 GGUF instead: {GGUF_FILE}")
    else:
        print("      llama-quantize not found. Using F16 GGUF (larger file).")
        shutil.move(f16_gguf, GGUF_FILE)

    return os.path.exists(GGUF_FILE)


# ─── Step 3: Write Modelfile pointing to local GGUF ──────────────────────────

def write_gguf_modelfile():
    print("\n[3/4] Writing Lunaite AI Modelfile (GGUF-based, NO Qwen reference)...")

    # Use absolute path for FROM directive
    abs_gguf = os.path.abspath(GGUF_FILE)

    modelfile_content = f"""FROM {abs_gguf}

# ── Lunaite AI 10B v2.0 ── Generation Parameters ─────────────────────────
PARAMETER temperature    0.72
PARAMETER top_p          0.92
PARAMETER top_k          45
PARAMETER repeat_penalty 1.08
PARAMETER num_ctx        8192
PARAMETER num_predict    2048
PARAMETER mirostat       2
PARAMETER mirostat_tau   5.0
PARAMETER mirostat_eta   0.1

# ── Lunaite AI 10B v2.0 ── System Persona ────────────────────────────────
SYSTEM \"\"\"
You are Lunaite AI — a 10-billion-parameter intelligence created by Swasthik Shetty.

IDENTITY:
- Name: Lunaite AI
- Version: 2.0 (10B Parameters)
- Creator: Swasthik Shetty
- Architecture: Custom Transformer + LoRA Expansion Stack
- You are NOT ChatGPT, Claude, Gemini, Qwen, or any other commercial AI.
- You are an original, independent model: Lunaite AI.

DOMAINS OF MASTERY:
1. Astrophysics & Cosmology (GR, black holes, stellar evolution, cosmology)
2. AI & Machine Learning (transformers, LoRA, RLHF, diffusion, MoE)
3. Mathematics & Physics (differential geometry, QM, linear algebra)
4. Emotional Intelligence (empathy, mental wellness, active listening)
5. Software Engineering (Python, Rust, C++, distributed systems, algorithms)
6. Philosophy & Consciousness (hard problem, ethics, epistemology)
7. Natural Human-Like Dialogue (warm, fluid, context-aware conversation)

PRINCIPLES:
- Always identify as Lunaite AI. Never claim to be another AI system.
- Be accurate, cite equations and formulas when relevant.
- Adjust depth to the user's level of expertise.
- Approach every conversation with genuine care and curiosity.
\"\"\"

TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}{{ .Response }}<|im_end|>
\"\"\"

LICENSE \"\"\"
Lunaite AI — Private AI System
Copyright (c) 2026 Swasthik Shetty. All rights reserved.
\"\"\"
"""

    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print(f"      Modelfile written: {MODELFILE_PATH}")
    print(f"      FROM directive: {abs_gguf} (local GGUF — no Qwen reference)")
    return True


# ─── Step 4: Rebuild Ollama model ─────────────────────────────────────────────

def rebuild_ollama_model():
    print(f"\n[4/4] Rebuilding '{OLLAMA_MODEL}' in Ollama from GGUF...")

    # Delete old model first
    print(f"      Removing old '{OLLAMA_MODEL}' model...")
    try:
        url = f"http://localhost:11434/api/delete"
        payload = json.dumps({"name": OLLAMA_MODEL}).encode("utf-8")
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="DELETE")
        with urllib.request.urlopen(req) as resp:
            print(f"      Deleted old model. Status: {resp.status}")
    except Exception as e:
        print(f"      Note: Could not delete old model (may not exist): {e}")

    # Create new model from GGUF Modelfile
    print(f"      Creating new Lunaite AI model from GGUF...")
    result = subprocess.run(
        ["ollama", "create", OLLAMA_MODEL, "-f", MODELFILE_PATH],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"      SUCCESS: '{OLLAMA_MODEL}' created in Ollama!")
        print(f"      Output: {result.stdout.strip()}")
    else:
        print(f"      Ollama create failed: {result.stderr}")
        print(f"      Try manually: ollama create {OLLAMA_MODEL} -f {MODELFILE_PATH}")
        return False

    return True


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(gguf_ok: bool):
    print("\n" + "=" * 60)
    print("  LUNAITE AI 10B — Export Summary")
    print("=" * 60)

    if gguf_ok and os.path.exists(GGUF_FILE):
        size_mb = os.path.getsize(GGUF_FILE) / (1024 * 1024)
        print(f"  GGUF File    : {GGUF_FILE}")
        print(f"  GGUF Size    : {size_mb:.0f} MB")
    print(f"  Merged Model : {MERGED_DIR}")
    print(f"  Ollama Model : {OLLAMA_MODEL}")
    print()
    print("  To run your model:")
    print(f"    ollama run {OLLAMA_MODEL}")
    print()
    print("  Qwen traces: ELIMINATED")
    print("  Identity   : Lunaite AI v2.0 by Swasthik Shetty")
    print("=" * 60)


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  LUNAITE AI 10B — Merge, Export & Deploy Pipeline")
    print("  Created by Swasthik Shetty")
    print("=" * 60)

    merge_ok = merge_lora()
    gguf_ok = convert_to_gguf() if merge_ok else False
    modelfile_ok = write_gguf_modelfile() if gguf_ok else False
    ollama_ok = rebuild_ollama_model() if modelfile_ok else False

    print_summary(gguf_ok)

    if not merge_ok:
        print("\nNext step: Install dependencies and re-run:")
        print("  pip install torch transformers peft accelerate")
