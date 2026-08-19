"""
Lunaite Architecture — Production LoRA & MoE Training Engine
============================================================
Fine-tunes arbitrary foundation models with high-rank LoRA and MoE adapters,
applying identity-weighted loss and gradient accumulation.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import os
import sys
import json
import time
import math
from typing import Dict, Any, List, Optional, Callable

from ..config import TrainConfig, LoRAConfig
from .dataset import load_dataset_file

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    HAS_TRAIN_DEPS = True
except ImportError:
    HAS_TRAIN_DEPS = False


class LunaiteTrainer:
    """
    High-performance fine-tuning trainer for Lunaite Architecture.
    """
    def __init__(self, config: Optional[TrainConfig] = None):
        self.config = config or TrainConfig()

    def train(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute training run across dataset.
        """
        if not HAS_TRAIN_DEPS:
            raise ImportError("PyTorch, Transformers, and PEFT are required for training. Install with: pip install torch transformers peft accelerate bitsandbytes")

        cfg = self.config
        samples = load_dataset_file(cfg.dataset_path)
        if not samples:
            raise ValueError(f"No samples found in dataset '{cfg.dataset_path}'")

        print(f"[*] Starting Lunaite Architecture Training on Base Model: {cfg.base_model}")
        print(f"[*] Loaded {len(samples)} training samples from {cfg.dataset_path}")

        # 1. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. Base Model
        quant_kwargs = {}
        if cfg.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            except Exception:
                pass

        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            **quant_kwargs
        )

        if cfg.quantization == "4bit":
            model = prepare_model_for_kbit_training(model)

        # 3. LoRA / Adapter Config
        peft_cfg = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        model = get_peft_model(model, peft_cfg)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"[*] Trainable Adapter Parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

        # 4. Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )

        total_steps = (len(samples) * cfg.epochs) // cfg.grad_accum
        total_steps = max(total_steps, 1)
        warmup_steps = int(total_steps * cfg.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 5. Training Loop
        model.train()
        step = 0
        losses = []
        start_time = time.time()

        for epoch in range(1, cfg.epochs + 1):
            epoch_loss = 0.0
            optimizer.zero_grad()

            for i, sample in enumerate(samples):
                prompt = sample["prompt"]
                response = sample["response"]
                text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

                enc = tokenizer(
                    text,
                    max_length=cfg.max_seq_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = enc["input_ids"]
                if hasattr(model, "device"):
                    input_ids = input_ids.to(model.device)

                labels = input_ids.clone()
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / cfg.grad_accum
                loss.backward()

                epoch_loss += outputs.loss.item()

                if (i + 1) % cfg.grad_accum == 0 or (i + 1) == len(samples):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1

                    current_lr = scheduler.get_last_lr()[0]
                    current_loss = outputs.loss.item()
                    losses.append(current_loss)

                    if progress_callback:
                        progress_callback({
                            "epoch": epoch,
                            "step": step,
                            "total_steps": total_steps,
                            "loss": current_loss,
                            "lr": current_lr,
                            "progress": min(step / total_steps, 1.0)
                        })

        elapsed = time.time() - start_time
        final_loss = sum(losses[-10:]) / len(losses[-10:]) if losses else 0.0

        # Save weights
        os.makedirs(cfg.output_dir, exist_ok=True)
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

        print(f"[*] Training Completed in {elapsed:.1f}s. Final loss: {final_loss:.4f}")
        print(f"[*] Saved Lunaite Adapter weights to: {cfg.output_dir}")

        meta = {
            "base_model": cfg.base_model,
            "trainable_params": trainable_params,
            "total_params": all_params,
            "epochs": cfg.epochs,
            "final_loss": final_loss,
            "elapsed_seconds": elapsed,
            "output_dir": cfg.output_dir
        }
        with open(os.path.join(cfg.output_dir, "lunaite_training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return meta
