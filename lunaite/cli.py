"""
Lunaite Architecture — Command Line Interface (CLI)
===================================================
Provides command line execution for Lunaite Architecture:
- lunaite run <model>           : Interactive chat session with live web search and tools
- lunaite train                 : Fine-tune model with LoRA / MoE adapters
- lunaite merge                 : Merge adapters into standalone foundation weights
- lunaite studio                : Launch the interactive Web Studio
- lunaite info                  : Display hardware telemetry and architectural diagnostics

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import sys
import argparse
from typing import Optional

from .config import LunaiteConfig
from .models.wrapper import wrap
from .agent.desktop import get_system_telemetry
from .core.architecture import calculate_architecture_parameters


def run_chat_cli(model_name: str = "lunaite-ai", deliberate: bool = False):
    """Start interactive terminal chat."""
    print(f"\033[96m\033[1m┌────────────────────────────────────────────────────────┐\033[0m")
    print(f"\033[96m\033[1m│  Lunaite — Interactive Assistant                       │\033[0m")
    print(f"\033[96m\033[1m│  Built by Swasthik Shetty (swasthik.mk3@gmail.com)     │\033[0m")
    print(f"\033[96m\033[1m└────────────────────────────────────────────────────────┘\033[0m")
    print(f"\033[90mModel: \033[97m{model_name}\033[90m | Reasoning Mode: \033[97m{'ON' if deliberate else 'OFF'}\033[90m\033[0m")
    print(f"\033[90mCommands: '/exit' to quit, '/clear' to reset, '/deliberate' to toggle reasoning.\033[0m\n")

    model = wrap(model_name)

    while True:
        try:
            user_input = input("\033[96m› \033[0m").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
                print("\033[90mGoodbye!\033[0m")
                break
            elif user_input.lower() == "/clear":
                model.clear_history()
                print("\033[93m[*] Conversation history cleared.\033[0m")
                continue
            elif user_input.lower() == "/deliberate":
                deliberate = not deliberate
                print(f"\033[93m[*] Deliberation mode set to: {deliberate}\033[0m")
                continue

            print("\033[95m● Lunaite AI:\033[0m ", end="", flush=True)
            response = model.generate(user_input, use_deliberation=deliberate, use_agent=True)
            print(f"{response}\n")

        except KeyboardInterrupt:
            print("\n\033[90mSession interrupted.\033[0m")
            break
        except Exception as e:
            print(f"\n\033[91m[Error]: {e}\033[0m\n")


def main():
    parser = argparse.ArgumentParser(
        prog="lunaite",
        description="Lunaite — Cognitive Architecture & Agent Runtime for ANY AI Model"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run interactive chat with any model (e.g. qwen2.5:7b, llama3.1:8b, gpt-4o)")
    run_parser.add_argument("model", nargs="?", default="qwen2.5:7b", help="Model name or tag")
    run_parser.add_argument("--deliberate", action="store_true", help="Enable multi-perspective cognitive deliberation")

    # Info command
    subparsers.add_parser("info", help="Show system telemetry and diagnostics")

    args = parser.parse_args()

    if args.command == "run" or args.command is None:
        model_name = getattr(args, "model", "qwen2.5:7b") or "qwen2.5:7b"
        deliberate = getattr(args, "deliberate", False)
        run_chat_cli(model_name=model_name, deliberate=deliberate)

    elif args.command == "info":
        stats = get_system_telemetry()
        print("=== Lunaite Diagnostics ===")
        print(f"Author: Swasthik Shetty <swasthik.mk3@gmail.com>")
        print(f"Repository: https://github.com/hallow-mk3/Lunaite")
        print(f"CPU Usage: {stats['cpu_percent']}%")
        print(f"RAM Usage: {stats['ram_used_gb']}GB / {stats['ram_total_gb']}GB ({stats['ram_percent']}%)")
        print(f"GPU: {stats['gpu_name']} ({stats['gpu_vram_used_gb']}GB VRAM)")
        print(f"Free Disk: {stats['disk_free_gb']}GB")


if __name__ == "__main__":
    main()
