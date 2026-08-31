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
C_UNDERLINE = "\033[4m"

# Modern 24-bit TrueColor Palette
CYAN_GLOW = "\033[38;2;125;211;252m"   # #7dd3fc electric glow
CYAN_BRIGHT = "\033[38;2;56;189;248m"  # #38bdf8 primary bright cyan
CYAN_MAIN = "\033[38;2;6;182;212m"    # #06b6d4 signature cyan
CYAN_DEEP = "\033[38;2;14;116;144m"   # #0e7490 subtle frame cyan
CYAN_BG = "\033[48;2;8;47;73m"        # #082f49 pill background

SLATE_LIGHT = "\033[38;2;226;232;240m"# #e2e8f0 crisp body text
SLATE = "\033[38;2;148;163;184m"      # #94a3b8 muted details
SLATE_DARK = "\033[38;2;71;85;105m"   # #475569 frame lines
WHITE_BOLD = "\033[38;2;255;255;255m\033[1m"

GREEN_ACCENT = "\033[38;2;52;211;153m" # #34d399 emerald status ok
AMBER_ACCENT = "\033[38;2;251;191;36m" # #fbbf24 amber deliberation
PURPLE_ACCENT = "\033[38;2;192;132;252m"# #c084fc violet accent


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
                                "desc": f"Local Ollama checkpoint ({full_name})"
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
                        "desc": f"Active Ollama instance ({name})"
                    })
    except Exception:
        pass

    # 2. Check Cloud API Keys
    if os.getenv("OPENAI_API_KEY"):
        api_models.append({"name": "gpt-4o-mini", "backend": "api", "desc": "OpenAI (Fast, High Efficiency)"})
        api_models.append({"name": "gpt-4o", "backend": "api", "desc": "OpenAI (Complex Tool Orchestration)"})

    if os.getenv("GROQ_API_KEY"):
        api_models.append({"name": "llama-3.3-70b-versatile", "backend": "api", "desc": "Groq (Ultra-Fast 70B)"})

    if os.getenv("DEEPSEEK_API_KEY"):
        api_models.append({"name": "deepseek-chat", "backend": "api", "desc": "DeepSeek V3 (High Reasoning)"})

    if os.getenv("OPENROUTER_API_KEY"):
        api_models.append({"name": "openrouter/auto", "backend": "api", "desc": "OpenRouter Auto Multi-Provider"})

    return local_models, api_models


def select_model_interactive() -> Tuple[str, Optional[str]]:
    """Present a clean interactive menu for the user to choose an available model."""
    local_models, api_models = discover_available_models()
    all_options = []

    print(f"\n{CYAN_MAIN}╭──────────────────────────────────────────────────────────────╮{C_RESET}")
    print(f"{CYAN_MAIN}│{C_RESET}  {WHITE_BOLD}✦ Lunaite Runtime{C_RESET} {SLATE}— Model Selection Matrix{C_RESET}              {CYAN_MAIN}│{C_RESET}")
    print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────────╯{C_RESET}\n")

    idx = 1
    if local_models:
        print(f"  {CYAN_GLOW}{C_BOLD}Local Backends (Found on device):{C_RESET}")
        for m in local_models:
            print(f"    {CYAN_MAIN}{idx:>2}.{C_RESET} {WHITE_BOLD}{m['name']:<24}{C_RESET} {SLATE}• {m['desc']}{C_RESET}")
            all_options.append(m)
            idx += 1
        print()

    if api_models:
        print(f"  {GREEN_ACCENT}{C_BOLD}Cloud Endpoints (Environment Key Detected):{C_RESET}")
        for m in api_models:
            print(f"    {CYAN_MAIN}{idx:>2}.{C_RESET} {WHITE_BOLD}{m['name']:<24}{C_RESET} {SLATE}• {m['desc']}{C_RESET}")
            all_options.append(m)
            idx += 1
        print()

    # Custom entry option
    print(f"  {PURPLE_ACCENT}{C_BOLD}Custom Model Backend:{C_RESET}")
    print(f"    {CYAN_MAIN}{idx:>2}.{C_RESET} {WHITE_BOLD}Enter custom model name or remote endpoint{C_RESET}\n")

    default_choice = "1" if all_options else str(idx)
    prompt_str = f"  {CYAN_BRIGHT}Select [1-{idx}]{C_RESET} {SLATE_DARK}(default: {default_choice}){C_RESET} {CYAN_MAIN}❯{C_RESET} "

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
            custom_name = input(f"\n  {CYAN_BRIGHT}Enter model tag (e.g. qwen2.5:7b, gpt-4o, llama3.1):{C_RESET} ").strip()
            if not custom_name:
                custom_name = "qwen2.5:7b"
            backend = "api" if custom_name.startswith(("gpt-", "claude-", "deepseek-")) else "ollama"
            return custom_name, backend

    backend = "api" if user_choice.startswith(("gpt-", "claude-", "deepseek-")) else "ollama"
    return user_choice, backend


# ─── INTERACTIVE CHAT RUNNER ─────────────────────────────────────────────────

def print_banner(model_name: str, deliberate: bool):
    """Render a clean, modern Claude Code-style header."""
    cwd = os.getcwd()
    if len(cwd) > 38:
        cwd = "..." + cwd[-35:]

    deliberate_badge = f"{GREEN_ACCENT}enabled{C_RESET}" if deliberate else f"{SLATE}disabled{C_RESET}"

    print(f"\n{CYAN_MAIN}╭──────────────────────────────────────────────────────────────╮{C_RESET}")
    print(f"{CYAN_MAIN}│{C_RESET}  {CYAN_GLOW}{C_BOLD}✦ LUNAITE{C_RESET} {SLATE}v3.0.0{C_RESET}  {SLATE_DARK}│{C_RESET}  Model: {WHITE_BOLD}{model_name}{C_RESET}")
    print(f"{CYAN_MAIN}│{C_RESET}  {SLATE_DARK}Workspace:{C_RESET} {SLATE}{cwd:<34}{C_RESET} {SLATE_DARK}│{C_RESET}  Deliberation: {deliberate_badge}")
    print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────────╯{C_RESET}")
    print(f"  {SLATE}Type {CYAN_MAIN}/tools{SLATE} for active capabilities, {CYAN_MAIN}/help{SLATE} for commands, {CYAN_MAIN}/exit{SLATE} to quit.{C_RESET}\n")


def render_tool_call(tool_name: str, tool_arg: str):
    """Display an inline modern tool badge with structured styling."""
    print(f"  {CYAN_MAIN}┌─{C_RESET} {CYAN_GLOW}{C_BOLD}Tool Executed:{C_RESET} {WHITE_BOLD}{tool_name}{C_RESET}")
    if tool_arg:
        print(f"  {CYAN_MAIN}│{C_RESET}  {SLATE}Arguments:{C_RESET} {SLATE_LIGHT}{tool_arg}{C_RESET}")
    print(f"  {CYAN_MAIN}└─{C_RESET} {GREEN_ACCENT}✓ Executing call...{C_RESET}\n")


def render_deliberation_status(stage: str, message: str):
    """Display deliberation thinking status."""
    print(f"  {AMBER_ACCENT}◈ {C_ITALIC}{message}{C_RESET}")


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

    last_web_topic = ""  # Track last search subject for pronoun follow-ups
    session_history = []  # (role, text) pairs for /history command
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
                print(f"\n{CYAN_MAIN}╭─────────────────── Available Commands ───────────────────╮{C_RESET}")
                print(f"  {CYAN_BRIGHT}/tools{C_RESET}       {SLATE_LIGHT}Inspect available tool registry & trigger patterns{C_RESET}")
                print(f"  {CYAN_BRIGHT}/history{C_RESET}     {SLATE_LIGHT}Review multi-turn conversation in current session{C_RESET}")
                print(f"  {CYAN_BRIGHT}/deliberate{C_RESET}  {SLATE_LIGHT}Toggle multi-perspective verification reasoning{C_RESET}")
                print(f"  {CYAN_BRIGHT}/clear{C_RESET}       {SLATE_LIGHT}Reset current conversation memory & buffer{C_RESET}")
                print(f"  {CYAN_BRIGHT}/info{C_RESET}        {SLATE_LIGHT}Show real-time hardware telemetry and GPU metrics{C_RESET}")
                print(f"  {CYAN_BRIGHT}/exit{C_RESET}        {SLATE_LIGHT}Gracefully terminate the current session{C_RESET}")
                print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────╯{C_RESET}\n")
                continue
            elif cmd_lower == "/history":
                if not session_history:
                    print(f"  {SLATE}No conversation history recorded in this session.{C_RESET}\n")
                else:
                    print(f"\n{CYAN_MAIN}╭─────────────────── Session History ──────────────────────╮{C_RESET}")
                    for i, (role, text) in enumerate(session_history, 1):
                        role_color = CYAN_GLOW if role == "You" else GREEN_ACCENT
                        prefix = f"{role_color}{C_BOLD}{role:<7}{C_RESET}"
                        snippet = text[:100].replace("\n", " ")
                        suffix = "..." if len(text) > 100 else ""
                        print(f"  {SLATE_DARK}{i:>2}.{C_RESET} {prefix} {SLATE_DARK}│{C_RESET} {SLATE_LIGHT}{snippet}{suffix}{C_RESET}")
                    print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────╯{C_RESET}\n")
                continue
            elif cmd_lower == "/tools":
                print(f"\n{CYAN_MAIN}╭────────────────── Registered Tool Library ────────────────╮{C_RESET}")
                print(f"  {CYAN_GLOW}✦ Web Search{C_RESET}    {SLATE_LIGHT}DuckDuckGo realtime web search{C_RESET}       {SLATE_DARK}['search for...']{C_RESET}")
                print(f"  {CYAN_GLOW}✦ Wikipedia{C_RESET}     {SLATE_LIGHT}Deep encyclopedic knowledge lookup{C_RESET}   {SLATE_DARK}['who was...']{C_RESET}")
                print(f"  {CYAN_GLOW}✦ Weather{C_RESET}       {SLATE_LIGHT}Global weather & temperature metrics{C_RESET} {SLATE_DARK}['weather in...']{C_RESET}")
                print(f"  {CYAN_GLOW}✦ Read File{C_RESET}     {SLATE_LIGHT}Inspect local file contents safely{C_RESET}   {SLATE_DARK}['read file...']{C_RESET}")
                print(f"  {CYAN_GLOW}✦ Write File{C_RESET}    {SLATE_LIGHT}Create or save workspace files{C_RESET}       {SLATE_DARK}['save file...']{C_RESET}")
                print(f"  {CYAN_GLOW}✦ Shell/CLI{C_RESET}     {SLATE_LIGHT}Execute PowerShell/terminal commands{C_RESET}{SLATE_DARK}['run command...']{C_RESET}")
                print(f"  {CYAN_GLOW}✦ Screenshot{C_RESET}    {SLATE_LIGHT}Capture high-res screen display{C_RESET}      {SLATE_DARK}['take screenshot']{C_RESET}")
                print(f"  {CYAN_GLOW}✦ Clipboard{C_RESET}     {SLATE_LIGHT}Read or inject into system clipboard{C_RESET}{SLATE_DARK}['read clipboard']{C_RESET}")
                print(f"  {CYAN_GLOW}✦ Telemetry{C_RESET}     {SLATE_LIGHT}CPU, RAM, GPU, and disk metrics{C_RESET}     {SLATE_DARK}['system vitals']{C_RESET}")
                print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────╯{C_RESET}\n")
                continue
            elif cmd_lower == "/info":
                stats = get_system_telemetry()
                print(f"\n{CYAN_MAIN}╭────────────────── System Diagnostics ────────────────────╮{C_RESET}")
                print(f"  {SLATE}CPU Utilization:{C_RESET} {WHITE_BOLD}{stats['cpu_percent']}%{C_RESET}   {SLATE}RAM:{C_RESET} {WHITE_BOLD}{stats['ram_used_gb']}/{stats['ram_total_gb']} GB{C_RESET} {SLATE_DARK}({stats['ram_percent']}%){C_RESET}")
                print(f"  {SLATE}GPU Accelerator:{C_RESET} {WHITE_BOLD}{stats['gpu_name']}{C_RESET} {SLATE_DARK}({stats['gpu_vram_used_gb']} GB VRAM){C_RESET}")
                print(f"  {SLATE}Storage Free:{C_RESET}    {WHITE_BOLD}{stats['disk_free_gb']} GB available{C_RESET}")
                print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────╯{C_RESET}\n")
                continue

            # Track user input in session history
            session_history.append(("You", user_input))

            # Response header
            print(f"\n{CYAN_GLOW}{C_BOLD}✦ Lunaite{C_RESET}")

            # Execution with tool awareness and deliberation callbacks
            memory_ctx = model.memory.get_context_summary() if model.memory else ""
            # Inject last known topic so follow-up pronouns ("it", "this") resolve correctly
            enriched_ctx = f"Q: {last_web_topic}" if last_web_topic else memory_ctx
            intent = model.agent.decide_tool(user_input, lambda p: model._raw_generate(p), context=enriched_ctx) if model.agent else None

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
                # Track subject for ALL tool calls so follow-up pronouns resolve correctly
                if intent[0] in ("web_search", "wiki", "weather", "time"):
                    last_web_topic = intent[1]

                memory_ctx = model.memory.get_context_summary() if model.memory else ""
                system_prompt = model.cognitive.get_system_prompt(memory_ctx) if model.cognitive else ""
                augmented_prompt = (
                    f"{system_prompt}\n\n"
                    f"[Tool Execution Result — {intent[0].upper()}]:\n"
                    f"{tool_output}\n\n"
                    f"User Query: {user_input}\n\n"
                    f"Context: You are Lunaite, a local AI agent running directly on the user's machine with full access to their local file system, clipboard, and system tools.\n"
                    f"IMPORTANT: Answer ONLY using the tool results above. "
                    f"Do NOT claim you cannot access local files or the clipboard — you can. "
                    f"Do NOT invent names, organizations, or facts not present in the results. "
                    f"If the clipboard or file contains a path, acknowledge what it is and offer to read it. "
                    f"If the specific fact is not in the results, say so and suggest an official source."
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
                session_history.append(("Lunaite", full_text))

                # Auto-detect file path in clipboard output and offer to read it
                if intent[0] == "clipboard_read":
                    import re as _re, os as _os
                    path_match = _re.search(r'([A-Za-z]:\\[^\n\r"<>|*?]{3,})', tool_output)
                    if path_match:
                        fpath = path_match.group(1).strip()
                        if _os.path.isfile(fpath):
                            ext = _os.path.splitext(fpath)[1].lower()
                            if ext in (".txt", ".md", ".py", ".json", ".csv", ".log", ".yaml", ".toml"):
                                print(f"{SLATE_DARK}↳ Auto-reading text file from clipboard...{C_RESET}")
                                file_contents = model.agent.execute_tool("read_file", fpath)
                                print(f"{SLATE}{file_contents[:2000]}{C_RESET}\n")
                            elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
                                print(f"{SLATE_DARK}↳ Image file in clipboard: {fpath}{C_RESET}")
                                print(f"{SLATE}   Say \"describe this image\" and I will analyze it.{C_RESET}\n")
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
                    session_history.append(("Lunaite", full_text))

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
