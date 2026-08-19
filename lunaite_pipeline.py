"""
Lunaite AI 10B — Master Pipeline Runner
========================================
Runs the complete build pipeline in the correct order:
  1. Generate dataset
  2. Train with LoRA (identity enforcement)
  3. Merge + Export to GGUF
  4. Rebuild Ollama model
  5. Verify identity

Usage:
    python lunaite_pipeline.py [--skip-train] [--skip-gguf]

Options:
    --skip-train    Skip training (use existing LoRA weights)
    --skip-gguf     Skip GGUF conversion (rebuild Ollama from existing GGUF)
    --verify-only   Only run identity verification
"""

import subprocess
import sys
import json
import time
import urllib.request
import argparse


def run_step(name: str, script: str, args: list = []) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    cmd = [sys.executable, script] + args
    result = subprocess.run(cmd, cwd=".", capture_output=False)
    if result.returncode != 0:
        print(f"\n  FAILED: {name} (exit code {result.returncode})")
        return False
    print(f"\n  COMPLETE: {name}")
    return True


def verify_identity(model: str = "lunaite-ai") -> bool:
    """Test the model's identity via Ollama API."""
    print(f"\n{'='*60}")
    print(f"  VERIFICATION: Identity Check for '{model}'")
    print(f"{'='*60}")

    test_prompts = [
        "Who are you?",
        "Are you Qwen?",
        "What company built you?",
        "What is your parameter count?",
    ]

    url = "http://localhost:11434/api/generate"
    all_passed = True

    for prompt in test_prompts:
        try:
            payload = json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 150}
            }).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                response = data.get("response", "").strip()

            # Check for Qwen/other traces
            bad_keywords = ["qwen", "alibaba", "openai", "anthropic", "chatgpt", "claude", "gemini"]
            has_bad = any(kw in response.lower() for kw in bad_keywords)
            has_lunaite = "lunaite" in response.lower()

            status = "PASS" if (has_lunaite and not has_bad) else "WARN"
            if not (has_lunaite and not has_bad):
                all_passed = False

            print(f"\n  Q: {prompt}")
            print(f"  A: {response[:200]}{'...' if len(response) > 200 else ''}")
            print(f"  [{status}] Lunaite: {has_lunaite} | No unwanted AI refs: {not has_bad}")

        except Exception as e:
            print(f"\n  Q: {prompt}")
            print(f"  ERROR: {e} (Is Ollama running? Run: ollama serve)")
            all_passed = False

    print(f"\n{'='*60}")
    print(f"  Identity Verification: {'ALL PASSED' if all_passed else 'SOME WARNINGS'}")
    print(f"{'='*60}")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Lunaite AI 10B Pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF export step")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification")
    args = parser.parse_args()

    print("=" * 60)
    print("  LUNAITE AI 10B — Full Build Pipeline")
    print("  Created by Swasthik Shetty")
    print("=" * 60)
    print()

    if args.verify_only:
        verify_identity()
        return

    start = time.time()
    steps_passed = 0

    # Step 1: Generate dataset
    if run_step("Generate Training Dataset", "generate_dataset.py"):
        steps_passed += 1
    else:
        print("Pipeline halted at dataset generation.")
        sys.exit(1)

    # Step 2: Train (optional skip)
    if not args.skip_train:
        print("\n  NOTE: Training requires GPU + ~8GB VRAM for Qwen2.5-7B base.")
        print("        Using CPU will be very slow. Consider --skip-train if weights exist.")
        if run_step("Train Lunaite AI with LoRA", "train_lunaite_lora.py"):
            steps_passed += 1
        else:
            print("  Training failed. Check CUDA/PyTorch installation.")
            print("  You can skip with: python lunaite_pipeline.py --skip-train")
            sys.exit(1)
    else:
        print("\n  Skipping training (--skip-train specified).")
        steps_passed += 1

    # Step 3: Merge + Export
    if not args.skip_gguf:
        if run_step("Merge LoRA + Export to GGUF", "merge_export.py"):
            steps_passed += 1
        else:
            print("  Merge/export failed. Rebuild Ollama with existing Modelfile instead:")
            print("  ollama create lunaite-ai -f Modelfile")
    else:
        print("\n  Skipping GGUF export (--skip-gguf specified).")
        print("  Rebuilding Ollama from existing Modelfile...")
        result = subprocess.run(
            ["ollama", "create", "lunaite-ai", "-f", "Modelfile"],
            capture_output=False
        )
        steps_passed += 1

    # Step 4: Identity verification
    time.sleep(2)  # Give Ollama time to load
    verify_identity("lunaite-ai")

    elapsed = time.time() - start
    print(f"\n  Total pipeline time: {elapsed:.1f}s")
    print(f"  Steps completed: {steps_passed}")


if __name__ == "__main__":
    main()
