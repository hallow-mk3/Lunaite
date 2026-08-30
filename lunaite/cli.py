"""
Lunaite Architecture — Modern Interactive CLI
=============================================
Claude Code-style intelligent terminal interface with rich cyan_bbt styling,
automatic model discovery, streaming generation, tool banners, and reasoning.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import sys
import os
import time
import argparse
import subprocess
import urllib.request
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import LunaiteConfig
from .models.wrapper import wrap
from .models.ollama import ensure_ollama_running
from .agent.desktop import get_system_telemetry


# ─── CYAN_BBT PALETTE & ANSI FORMATTING ──────────────────────────────────────
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_ITALIC = "\033[3m"

CYAN_BRIGHT = "\033[38;2;56;189;248m"    # #38bdf8 bright cyan
CYAN_MAIN = "\033[38;2;6;182;212m"      # #06b6d4 core cyan
CYAN_DEEP = "\033[38;2;14;116;144m"     # #0e7490 deep cyan accent
CYAN_BG = "\033[48;2;8;47;73m"          # #082f49 dark cyan background
SLATE = "\033[38;2;148;163;184m"        # #94a3b8 muted text
SLATE_DARK = "\033[38;2;71;85;105m"     # #475569 borders/separators
WHITE_BOLD = "\033[38;2;248;250;252m\033[1m"
GREEN_ACCENT = "\033[38;2;52;211;153m"  # #34d399 status ok
AMBER_ACCENT = "\033[38;2;251;191;36m"  # #fbbf24 thinking indicator


# ─── MODEL DISCOVERY ─────────────────────────────────────────────────────────

def discover_available_models() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Scan local machine and environment variables to find all usable AI models."""
    local_models: List[Dict[str, str]] = []
    api_models: List[Dict[str, str]] = []

    # 1. Check local Ollama models on disk (Windows / macOS / Linux)
    home = Path.home()
    ollama_manifest_dir = home / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
    if ollama_manifest_dir.exists():
        try:
            for model_dir in ollama_manifest_dir.iterdir():
                if model_dir.is_dir():
                    model_base = model_dir.name
                    for tag_file in model_dir.iterdir():
                        if tag_file.is_file():
                            tag = tag_file.name
                            full_name = f"{model_base}:{tag}" if tag != "latest" else model_base
                            local_models.append({
                                "name": full_name,
                                "backend": "ollama",
                                "desc": f"Local Ollama ({full_name})"
                            })
        except Exception:
            pass

    # Fallback to querying active Ollama server if running
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "Lunaite"})
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            data = json.loads(resp.read().decode())
            active_names = [m["name"] for m in data.get("models", [])]
            for name in active_names:
                if not any(m["name"] == name for m in local_models):
                    local_models.append({
                        "name": name,
                        "backend": "ollama",
                        "desc": f"Local Ollama ({name})"
                    })
    except Exception:
        pass

    # 2. Check Cloud API Keys
    if os.getenv("OPENAI_API_KEY"):
        api_models.append({"name": "gpt-4o-mini", "backend": "api", "desc": "OpenAI (Fast & Cheap)"})
        api_models.append({"name": "gpt-4o", "backend": "api", "desc": "OpenAI (High Intelligence)"})

    if os.getenv("GROQ_API_KEY"):
        api_models.append({"name": "llama-3.3-70b-versatile", "backend": "api", "desc": "Groq (Ultra-Fast 70B)"})

    if os.getenv("DEEPSEEK_API_KEY"):
        api_models.append({"name": "deepseek-chat", "backend": "api", "desc": "DeepSeek V3"})

    if os.getenv("OPENROUTER_API_KEY"):
        api_models.append({"name": "openrouter/auto", "backend": "api", "desc": "OpenRouter Auto Router"})

    return local_models, api_models


def select_model_interactive() -> Tuple[str, Optional[str]]:
    """Present a clean interactive menu for the user to choose an available model."""
    local_models, api_models = discover_available_models()
    all_options = []

    print(f"\n{CYAN_MAIN}{C_BOLD}● Lunaite — Select an AI Model{C_RESET}")
    print(f"{SLATE_DARK}------------------------------------------------------------{C_RESET}")

    idx = 1
    if local_models:
        print(f"  {CYAN_BRIGHT}{C_BOLD}Local Models (Found on system):{C_RESET}")
        for m in local_models:
            print(f"    {CYAN_MAIN}[{idx}]{C_RESET} {WHITE_BOLD}{m['name']:<20}{C_RESET} {SLATE}{m['desc']}{C_RESET}")
            all_options.append(m)
            idx += 1
        print()

    if api_models:
        print(f"  {GREEN_ACCENT}{C_BOLD}Cloud API Models (Key detected):{C_RESET}")
        for m in api_models:
            print(f"    {CYAN_MAIN}[{idx}]{C_RESET} {WHITE_BOLD}{m['name']:<26}{C_RESET} {SLATE}{m['desc']}{C_RESET}")
            all_options.append(m)
            idx += 1
        print()

    # Custom entry option
    print(f"  {AMBER_ACCENT}{C_BOLD}Custom Option:{C_RESET}")
    print(f"    {CYAN_MAIN}[{idx}]{C_RESET} {WHITE_BOLD}Enter custom model name or OpenAI API key{C_RESET}\n")

    default_choice = "1" if all_options else str(idx)
    prompt_str = f"{CYAN_BRIGHT}Select [1-{idx}]{C_RESET} {SLATE}(default {default_choice}):{C_RESET} "

    try:
        user_choice = input(prompt_str).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

    if not user_choice:
        user_choice = default_choice

    if user_choice.isdigit():
        choice_num = int(user_choice)
        if 1 <= choice_num <= len(all_options):
            selected = all_options[choice_num - 1]
            return selected["name"], selected["backend"]
        elif choice_num == idx:
            custom_name = input(f"{CYAN_BRIGHT}Enter model name (e.g. qwen2.5:7b, gpt-4o, llama3.1):{C_RESET} ").strip()
            if not custom_name:
                custom_name = "qwen2.5:7b"
            backend = "api" if custom_name.startswith(("gpt-", "claude-", "deepseek-")) else "ollama"
            return custom_name, backend

    # If the user typed the name directly
    backend = "api" if user_choice.startswith(("gpt-", "claude-", "deepseek-")) else "ollama"
    return user_choice, backend


# ─── INTERACTIVE CHAT RUNNER ─────────────────────────────────────────────────

def print_banner(model_name: str, deliberate: bool):
    """Render a clean Claude Code-inspired header in cyan_bbt."""
    cwd = os.getcwd()
    if len(cwd) > 40:
        cwd = "..." + cwd[-37:]

    print(f"\n{CYAN_MAIN}{C_BOLD}● Lunaite{C_RESET} {SLATE}v3.0.0{C_RESET} {SLATE_DARK}({model_name}){C_RESET}")
    print(f"{SLATE_DARK}dir:{C_RESET} {SLATE}{cwd}{C_RESET}")
    status_mode = f"{CYAN_BRIGHT}on{C_RESET}" if deliberate else f"{SLATE}off{C_RESET}"
    print(f"{SLATE_DARK}deliberation:{C_RESET} {status_mode} {SLATE_DARK}• type {CYAN_MAIN}/help{SLATE_DARK} for commands{C_RESET}\n")


def render_tool_call(tool_name: str, tool_arg: str):
    """Display an inline Claude Code style tool badge."""
    print(f"\n{CYAN_DEEP}┌─ {CYAN_BRIGHT}{C_BOLD}Tool: {tool_name}{C_RESET} {SLATE_DARK}({tool_arg}){C_RESET}")
    print(f"{CYAN_DEEP}└─ Running query...{C_RESET}\n")


def render_deliberation_status(stage: str, message: str):
    """Display deliberation thinking status."""
    print(f"  {AMBER_ACCENT}⟳ {C_ITALIC}{message}{C_RESET}")


def run_chat_cli(
    model_name: Optional[str] = None,
    backend: Optional[str] = None,
    deliberate: bool = False
):
    """Start interactive Claude Code-style chat session."""
    # If no model specified, auto-discover and prompt
    if not model_name or model_name == "auto":
        model_name, backend = select_model_interactive()

    # Ensure Ollama server is active if using local backend
    if backend == "ollama" or backend is None:
        ensure_ollama_running()

    print_banner(model_name=model_name, deliberate=deliberate)
    model = wrap(model_name, backend=backend)

    while True:
        try:
            prompt_symbol = f"{CYAN_BRIGHT}{C_BOLD}❯{C_RESET} "
            user_input = input(prompt_symbol).strip()
            if not user_input:
                continue

            cmd_lower = user_input.lower()
            if cmd_lower in ["/exit", "/quit", "exit", "quit", ":q"]:
                print(f"\n{SLATE}Session ended. Farewell.{C_RESET}\n")
                break
            elif cmd_lower in ["/clear", "/reset"]:
                model.clear_history()
                print(f"{SLATE_DARK}Conversation context reset.{C_RESET}\n")
                continue
            elif cmd_lower == "/deliberate":
                deliberate = not deliberate
                mode_str = f"{CYAN_BRIGHT}enabled{C_RESET}" if deliberate else f"{SLATE}disabled{C_RESET}"
                print(f"{SLATE_DARK}Deliberation mode {mode_str}.{C_RESET}\n")
                continue
            elif cmd_lower in ["/help", "/?"]:
                print(f"\n{WHITE_BOLD}Lunaite Commands:{C_RESET}")
                print(f"  {CYAN_MAIN}/tools{C_RESET}       - List all available tools & triggers")
                print(f"  {CYAN_MAIN}/deliberate{C_RESET}  - Toggle multi-perspective reasoning")
                print(f"  {CYAN_MAIN}/clear{C_RESET}       - Clear conversation memory buffer")
                print(f"  {CYAN_MAIN}/info{C_RESET}        - Show live hardware and system vitals")
                print(f"  {CYAN_MAIN}/exit{C_RESET}        - Exit session\n")
                continue
            elif cmd_lower == "/tools":
                print(f"\n{WHITE_BOLD}Available Lunaite Tools:{C_RESET}")
                print(f"  {CYAN_BRIGHT}● Web Search{C_RESET}    {SLATE}Live DuckDuckGo search{C_RESET}       {SLATE_DARK}(e.g. 'search for latest AI news'){C_RESET}")
                print(f"  {CYAN_BRIGHT}● Wikipedia{C_RESET}     {SLATE}Article knowledge lookup{C_RESET}     {SLATE_DARK}(e.g. 'who was Alan Turing?'){C_RESET}")
                print(f"  {CYAN_BRIGHT}● Weather{C_RESET}       {SLATE}Global forecast by city{C_RESET}      {SLATE_DARK}(e.g. 'weather in Tokyo'){C_RESET}")
                print(f"  {CYAN_BRIGHT}● Read File{C_RESET}     {SLATE}Inspect file contents{C_RESET}        {SLATE_DARK}(e.g. 'read the file README.md'){C_RESET}")
                print(f"  {CYAN_BRIGHT}● Write File{C_RESET}    {SLATE}Create or save file{C_RESET}         {SLATE_DARK}(e.g. 'save file notes.txt : content'){C_RESET}")
                print(f"  {CYAN_BRIGHT}● Shell/CLI{C_RESET}     {SLATE}Run PowerShell/shell cmd{C_RESET}     {SLATE_DARK}(e.g. 'run command: git status'){C_RESET}")
                print(f"  {CYAN_BRIGHT}● Screenshot{C_RESET}    {SLATE}Capture screen image{C_RESET}         {SLATE_DARK}(e.g. 'take a screenshot'){C_RESET}")
                print(f"  {CYAN_BRIGHT}● Clipboard{C_RESET}     {SLATE}Read system clipboard{C_RESET}        {SLATE_DARK}(e.g. 'read clipboard'){C_RESET}")
                print(f"  {CYAN_BRIGHT}● Telemetry{C_RESET}     {SLATE}CPU, RAM, GPU, Disk stats{C_RESET}    {SLATE_DARK}(e.g. 'system vitals' or /info){C_RESET}\n")
                continue
            elif cmd_lower == "/info":
                stats = get_system_telemetry()
                print(f"\n{CYAN_MAIN}{C_BOLD}System Diagnostics:{C_RESET}")
                print(f"  {SLATE}CPU:{C_RESET} {stats['cpu_percent']}%  {SLATE}RAM:{C_RESET} {stats['ram_used_gb']}/{stats['ram_total_gb']} GB ({stats['ram_percent']}%)")
                print(f"  {SLATE}GPU:{C_RESET} {stats['gpu_name']} ({stats['gpu_vram_used_gb']} GB VRAM)")
                print(f"  {SLATE}Disk:{C_RESET} {stats['disk_free_gb']} GB free\n")
                continue

            # Response header
            print(f"\n{CYAN_MAIN}{C_BOLD}● Lunaite{C_RESET}")

            # Execution with tool awareness and deliberation callbacks
            memory_ctx = model.memory.get_context_summary() if model.memory else ""
            intent = model.agent.decide_tool(user_input, lambda p: model._raw_generate(p), context=memory_ctx) if model.agent else None

            if deliberate and model.cognitive:
                response = model.cognitive.deliberate(
                    user_input,
                    lambda p: model._raw_generate(p),
                    progress_callback=render_deliberation_status
                )
                print(f"\n{response}\n")
            elif intent:
                render_tool_call(intent[0], intent[1])
                tool_output = model.agent.execute_tool(intent[0], intent[1])

                memory_ctx = model.memory.get_context_summary() if model.memory else ""
                system_prompt = model.cognitive.get_system_prompt(memory_ctx) if model.cognitive else ""
                augmented_prompt = (
                    f"{system_prompt}\n\n"
                    f"[Tool Execution Result — {intent[0].upper()}]:\n"
                    f"{tool_output}\n\n"
                    f"User Query: {user_input}\n\n"
                    f"Based on the tool results above, provide an accurate, clear, and direct answer:"
                )

                full_tokens = []
                for chunk in model._raw_stream_generate(augmented_prompt):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    full_tokens.append(chunk)
                print("\n")

                full_text = "".join(full_tokens).strip()
                if model.memory and len(full_text.split()) > 8:
                    model.memory.add_insight(f"Q: {user_input[:50]} -> A: {full_text[:70]}")
            else:
                full_tokens = []
                try:
                    memory_ctx = model.memory.get_context_summary() if model.memory else ""
                    system_prompt = model.cognitive.get_system_prompt(memory_ctx) if model.cognitive else ""
                    full_prompt = f"{system_prompt}\n\nUser: {user_input}\nLunaite AI:" if system_prompt else user_input

                    for chunk in model._raw_stream_generate(full_prompt):
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        full_tokens.append(chunk)
                    print("\n")

                    full_text = "".join(full_tokens).strip()
                    if model.memory and len(full_text.split()) > 8:
                        model.memory.add_insight(f"Q: {user_input[:50]} -> A: {full_text[:70]}")
                except Exception:
                    response = model.generate(user_input, use_deliberation=False, use_agent=False)
                    print(f"{response}\n")

        except KeyboardInterrupt:
            print(f"\n{SLATE}Interrupted.{C_RESET}\n")
            break
        except Exception as e:
            print(f"\n\033[91m[Error]: {e}{C_RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="lunaite",
        description="Lunaite — Intelligent Agent & Tool Selection Runtime for ANY AI Model"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run interactive chat with any model")
    run_parser.add_argument("model", nargs="?", default=None, help="Optional model name (auto-prompts if omitted)")
    run_parser.add_argument("--deliberate", action="store_true", help="Enable multi-perspective cognitive deliberation")
    run_parser.add_argument("--backend", default=None, help="Backend: ollama, api, or huggingface")

    # Info command
    subparsers.add_parser("info", help="Show system telemetry and diagnostics")

    args = parser.parse_args()

    if args.command == "run" or args.command is None:
        model_name = getattr(args, "model", None)
        deliberate = getattr(args, "deliberate", False)
        backend = getattr(args, "backend", None)
        run_chat_cli(model_name=model_name, backend=backend, deliberate=deliberate)

    elif args.command == "info":
        stats = get_system_telemetry()
        print(f"\n{CYAN_MAIN}{C_BOLD}Lunaite Diagnostics:{C_RESET}")
        print(f"  Author: Swasthik Shetty <swasthik.mk3@gmail.com>")
        print(f"  Repository: https://github.com/hallow-mk3/Lunaite")
        print(f"  CPU Usage: {stats['cpu_percent']}%")
        print(f"  RAM Usage: {stats['ram_used_gb']}GB / {stats['ram_total_gb']}GB ({stats['ram_percent']}%)")
        print(f"  GPU: {stats['gpu_name']} ({stats['gpu_vram_used_gb']}GB VRAM)")
        print(f"  Free Disk: {stats['disk_free_gb']}GB\n")


if __name__ == "__main__":
    main()
