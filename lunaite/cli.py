"""
Lunaite Architecture — Modern Interactive CLI & Developer Suite
================================================================
Claude Code-style intelligent terminal interface with rich cyan_bbt styling,
automatic model discovery, streaming generation, tool execution, session branching,
context visualization, memory compaction, and backgrounding.

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
import uuid
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
RED_ACCENT = "\033[38;2;248;113;113m"   # #f87171 alert red


# ─── SESSION STORE & PERSISTENCE ──────────────────────────────────────────────

def get_session_dir() -> Path:
    """Return ~/.lunaite/sessions/ directory, creating it if needed."""
    session_dir = Path.home() / ".lunaite" / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_session_to_disk(session_data: Dict[str, Any]) -> Path:
    """Save session data JSON to ~/.lunaite/sessions/<session_id>.json."""
    s_id = session_data.get("session_id", time.strftime("session_%Y%m%d_%H%M%S"))
    target = get_session_dir() / f"{s_id}.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)
    return target


def list_saved_sessions() -> List[Dict[str, Any]]:
    """List all saved sessions sorted newest first."""
    s_dir = get_session_dir()
    results = []
    for f in sorted(s_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                data["_file"] = str(f)
                data["_mtime"] = os.path.getmtime(f)
                results.append(data)
        except Exception:
            pass
    return results


def load_session_from_disk(session_id: str) -> Optional[Dict[str, Any]]:
    """Load a session by exact ID or filename prefix."""
    s_dir = get_session_dir()
    # Try exact match
    exact = s_dir / f"{session_id}.json"
    if exact.exists():
        with open(exact, "r", encoding="utf-8") as fp:
            return json.load(fp)
    # Try prefix match
    for f in s_dir.glob("*.json"):
        if session_id.lower() in f.stem.lower():
            with open(f, "r", encoding="utf-8") as fp:
                return json.load(fp)
    return None


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

    return local_models, api_models


def select_model_interactive() -> Tuple[str, str]:
    """Display interactive numbered selection menu for discovered models."""
    local_models, api_models = discover_available_models()

    print(f"\n{CYAN_MAIN}╭────────────────── Select AI Model Backend ───────────────────╮{C_RESET}")

    all_options: List[Dict[str, str]] = []
    idx = 1

    if local_models:
        print(f"  {CYAN_BRIGHT}{C_BOLD}Local Models (Found on system):{C_RESET}")
        for m in local_models:
            print(f"    {CYAN_MAIN}[{idx}]{C_RESET} {WHITE_BOLD}{m['name']:<22}{C_RESET} {SLATE}{m['desc']}{C_RESET}")
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
    print(f"    {CYAN_MAIN}[{idx}]{C_RESET} {WHITE_BOLD}Enter custom model name or OpenAI API key{C_RESET}")
    print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────────╯{C_RESET}\n")

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
            custom_name = input(f"{CYAN_MAIN}Enter model name (e.g. qwen3:8b, mistral, gpt-4o):{C_RESET} ").strip()
            if not custom_name:
                custom_name = "lunaite-ai"
            backend = "api" if custom_name.startswith(("gpt-", "claude-")) else "ollama"
            return custom_name, backend

    return "lunaite-ai", "ollama"


# ─── BANNER & DISPLAY HELPERS ────────────────────────────────────────────────

def print_banner(model_name: str, deliberate: bool = False, branch: str = "main", cwd: Optional[str] = None):
    """Render modern Cyan TrueColor header banner."""
    cwd = cwd or os.getcwd()
    if len(cwd) > 34:
        cwd = "..." + cwd[-31:]

    deliberate_badge = f"{GREEN_ACCENT}active (multi-agent){C_RESET}" if deliberate else f"{SLATE}off{C_RESET}"

    print(f"\n{CYAN_MAIN}╭──────────────────────────────────────────────────────────────╮{C_RESET}")
    print(f"{CYAN_MAIN}│{C_RESET}  {CYAN_GLOW}{C_BOLD}✦ LUNAITE{C_RESET} {SLATE}v3.0.0{C_RESET}  {SLATE_DARK}│{C_RESET}  Model: {WHITE_BOLD}{model_name:<18}{C_RESET} {SLATE_DARK}Branch:{C_RESET} {CYAN_BRIGHT}{branch}{C_RESET}")
    print(f"{CYAN_MAIN}│{C_RESET}  {SLATE_DARK}Workspace:{C_RESET} {SLATE}{cwd:<34}{C_RESET} {SLATE_DARK}│{C_RESET}  Deliberation: {deliberate_badge}")
    print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────────╯{C_RESET}")
    print(f"  {SLATE}Type {CYAN_MAIN}/tools{SLATE} for registry, {CYAN_MAIN}/help{SLATE} for all commands, {CYAN_MAIN}/exit{SLATE} to quit.{C_RESET}\n")


def render_tool_call(tool_name: str, tool_arg: str):
    """Display an inline modern tool badge with structured styling."""
    print(f"  {CYAN_MAIN}┌─{C_RESET} {CYAN_GLOW}{C_BOLD}Tool Executed:{C_RESET} {WHITE_BOLD}{tool_name}{C_RESET}")
    if tool_arg:
        print(f"  {CYAN_MAIN}│{C_RESET}  {SLATE}Arguments:{C_RESET} {SLATE_LIGHT}{tool_arg}{C_RESET}")
    print(f"  {CYAN_MAIN}└─{C_RESET} {GREEN_ACCENT}✓ Executing call...{C_RESET}\n")


def render_deliberation_status(stage: str, message: str):
    """Display deliberation thinking status."""
    print(f"  {AMBER_ACCENT}◈ {C_ITALIC}{message}{C_RESET}")


def render_context_grid(
    session_history: List[Tuple[str, str]],
    model_name: str,
    active_dirs: List[str],
    current_branch: str,
    memory_ctx: str,
    max_context_tokens: int = 8192,
    autocompact_pct: int = 80
):
    """Visualize context buffer breakdown as a colored visual grid."""
    # Rough token estimation (1 word ≈ 1.33 tokens)
    persona_tokens = 380
    tools_tokens = 450
    memory_tokens = int(len(memory_ctx.split()) * 1.33)
    history_words = sum(len(txt.split()) for _, txt in session_history)
    history_tokens = int(history_words * 1.33)

    total_tokens = persona_tokens + tools_tokens + memory_tokens + history_tokens
    fill_pct = min(100.0, (total_tokens / max_context_tokens) * 100.0)

    # 24-char visual progress bar
    bar_len = 24
    filled = int((fill_pct / 100.0) * bar_len)
    if fill_pct < 60:
        bar_color = GREEN_ACCENT
    elif fill_pct < autocompact_pct:
        bar_color = AMBER_ACCENT
    else:
        bar_color = RED_ACCENT

    bar_str = f"{bar_color}{'█' * filled}{SLATE_DARK}{'░' * (bar_len - filled)}{C_RESET}"

    print(f"\n{CYAN_MAIN}╭───────────────────── Context Utilization ─────────────────────╮{C_RESET}")
    print(f"  {SLATE}Context Buffer:{C_RESET}  [{bar_str}] {WHITE_BOLD}{total_tokens:,}{C_RESET} / {max_context_tokens:,} tokens {bar_color}({fill_pct:.1f}%){C_RESET}")
    print(f"  {SLATE}Auto-Compact:{C_RESET}    {AMBER_ACCENT}Triggers at {autocompact_pct}% ({int(max_context_tokens * autocompact_pct / 100):,} tokens){C_RESET}")
    print()
    print(f"  {CYAN_BRIGHT}▣ System Persona:{C_RESET}  {SLATE_LIGHT}{persona_tokens:>5} tok{C_RESET} {SLATE_DARK}({persona_tokens/max_context_tokens*100:4.1f}%){C_RESET}   {CYAN_BRIGHT}▣ Tools Registry:{C_RESET} {SLATE_LIGHT}{tools_tokens:>5} tok{C_RESET} {SLATE_DARK}({tools_tokens/max_context_tokens*100:4.1f}%){C_RESET}")
    print(f"  {CYAN_BRIGHT}▣ Memory Bank:{C_RESET}     {SLATE_LIGHT}{memory_tokens:>5} tok{C_RESET} {SLATE_DARK}({memory_tokens/max_context_tokens*100:4.1f}%){C_RESET}   {CYAN_BRIGHT}▣ History Turns:{C_RESET}  {SLATE_LIGHT}{history_tokens:>5} tok{C_RESET} {SLATE_DARK}({history_tokens/max_context_tokens*100:4.1f}%){C_RESET}")
    print()
    print(f"  {CYAN_GLOW}📁 Active Workspaces ({len(active_dirs)}):{C_RESET}")
    for idx, d in enumerate(active_dirs, 1):
        tag = f"{GREEN_ACCENT}(primary){C_RESET}" if idx == 1 else ""
        print(f"     {SLATE_DARK}{idx}.{C_RESET} {SLATE_LIGHT}{d}{C_RESET} {tag}")
    print(f"  {PURPLE_ACCENT}🌿 Active Branch:{C_RESET}    {WHITE_BOLD}{current_branch}{C_RESET}  {SLATE_DARK}• Total Turns: {len(session_history)}{C_RESET}")
    print(f"{CYAN_MAIN}╰───────────────────────────────────────────────────────────────╯{C_RESET}\n")


# ─── INTERACTIVE CHAT RUNNER ─────────────────────────────────────────────────

def run_chat_cli(
    model_name: Optional[str] = None,
    backend: Optional[str] = None,
    deliberate: bool = False,
    resume_id: Optional[str] = None
):
    """Start interactive Claude Code-style chat session with developer command suite."""
    # Handle session resume
    loaded_session = None
    if resume_id:
        loaded_session = load_session_from_disk(resume_id)
        if loaded_session:
            model_name = loaded_session.get("model_name", model_name)
            backend = loaded_session.get("backend", backend)
            print(f"  {GREEN_ACCENT}✓ Restored session '{resume_id}' from disk.{C_RESET}")

    # If no model specified, auto-discover and prompt
    if not model_name or model_name == "auto":
        model_name, backend = select_model_interactive()

    # Ensure Ollama server is active if using local backend
    if backend == "ollama" or backend is None:
        ensure_ollama_running()

    # Session State
    session_id = loaded_session.get("session_id", time.strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:6]) if loaded_session else (time.strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:6])
    current_branch = loaded_session.get("branch", "main") if loaded_session else "main"
    active_dirs = loaded_session.get("active_dirs", [os.getcwd()]) if loaded_session else [os.getcwd()]
    autocompact_pct = 80
    max_context_tokens = 8192
    last_web_topic = ""
    session_history: List[Tuple[str, str]] = [(r, t) for r, t in loaded_session.get("session_history", [])] if loaded_session else []

    print_banner(model_name=model_name, deliberate=deliberate, branch=current_branch, cwd=active_dirs[0])
    model = wrap(model_name, backend=backend)

    while True:
        try:
            prompt_symbol = f"{CYAN_BRIGHT}{C_BOLD}❯{C_RESET} "
            user_input = input(prompt_symbol).strip()
            if not user_input:
                continue

            cmd_lower = user_input.lower().strip()
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            # ─────────────────────────────────────────────────────────────
            # 1. /exit & /quit
            # ─────────────────────────────────────────────────────────────
            if cmd in ["/exit", "/quit", "exit", "quit", ":q"]:
                # Auto-save session on exit
                save_session_to_disk({
                    "session_id": session_id,
                    "model_name": model_name,
                    "backend": backend,
                    "branch": current_branch,
                    "active_dirs": active_dirs,
                    "session_history": session_history,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                print(f"\n{SLATE}Session ended. State saved ({session_id}). Farewell.{C_RESET}\n")
                break

            # ─────────────────────────────────────────────────────────────
            # 2. /add-dir — Add working directory
            # ─────────────────────────────────────────────────────────────
            elif cmd == "/add-dir":
                if not arg:
                    print(f"\n{CYAN_MAIN}╭────────────────── Active Working Directories ────────────────╮{C_RESET}")
                    for idx, d in enumerate(active_dirs, 1):
                        tag = f"{GREEN_ACCENT}(primary){C_RESET}" if idx == 1 else ""
                        print(f"  {CYAN_BRIGHT}[{idx}]{C_RESET} {SLATE_LIGHT}{d}{C_RESET} {tag}")
                    print(f"\n  {SLATE}Usage: {CYAN_MAIN}/add-dir <path>{SLATE} to register another directory.{C_RESET}")
                    print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────────╯{C_RESET}\n")
                else:
                    target_dir = os.path.abspath(arg)
                    if not os.path.isdir(target_dir):
                        print(f"  {RED_ACCENT}Directory not found:{C_RESET} {target_dir}\n")
                    elif target_dir in active_dirs:
                        print(f"  {SLATE}Directory already in active workspaces:{C_RESET} {target_dir}\n")
                    else:
                        active_dirs.append(target_dir)
                        print(f"  {GREEN_ACCENT}✓ Added workspace:{C_RESET} {target_dir}")
                        print(f"  {SLATE}Active workspaces count:{C_RESET} {len(active_dirs)}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 3. /cd — Change primary working directory
            # ─────────────────────────────────────────────────────────────
            elif cmd == "/cd":
                if not arg:
                    print(f"  {SLATE}Current Directory:{C_RESET} {WHITE_BOLD}{os.getcwd()}{C_RESET}\n")
                else:
                    try:
                        target = os.path.abspath(os.path.expanduser(arg))
                        os.chdir(target)
                        active_dirs[0] = target
                        print(f"  {GREEN_ACCENT}✓ Changed working directory to:{C_RESET} {target}\n")
                    except Exception as e:
                        print(f"  {RED_ACCENT}Failed to change directory:{C_RESET} {e}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 4. /autocompact — Set context full threshold before summarization
            # ─────────────────────────────────────────────────────────────
            elif cmd == "/autocompact":
                if not arg:
                    print(f"\n  {CYAN_BRIGHT}Auto-Compact Status:{C_RESET} {GREEN_ACCENT}enabled{C_RESET}")
                    print(f"  {SLATE}Trigger Threshold:{C_RESET} {WHITE_BOLD}{autocompact_pct}%{C_RESET} of context window ({int(max_context_tokens * autocompact_pct / 100):,} tokens)")
                    print(f"  {SLATE}Set new threshold: {CYAN_MAIN}/autocompact 75%{SLATE} or {CYAN_MAIN}/autocompact 5000{C_RESET}\n")
                else:
                    clean_val = arg.replace("%", "").strip()
                    if clean_val.isdigit():
                        num = int(clean_val)
                        if num <= 100:
                            autocompact_pct = num
                            print(f"  {GREEN_ACCENT}✓ Auto-compact threshold set to {autocompact_pct}% of context window.{C_RESET}\n")
                        else:
                            autocompact_pct = min(95, max(20, int((num / max_context_tokens) * 100)))
                            print(f"  {GREEN_ACCENT}✓ Auto-compact set for {num} tokens ({autocompact_pct}%).{C_RESET}\n")
                    elif clean_val in ["off", "disable", "disabled"]:
                        autocompact_pct = 999
                        print(f"  {AMBER_ACCENT}Auto-compact disabled.{C_RESET}\n")
                    elif clean_val in ["on", "enable", "enabled"]:
                        autocompact_pct = 80
                        print(f"  {GREEN_ACCENT}✓ Auto-compact enabled (default 80%).{C_RESET}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 5. /background — Save state and free terminal
            # ─────────────────────────────────────────────────────────────
            elif cmd == "/background":
                saved_path = save_session_to_disk({
                    "session_id": session_id,
                    "model_name": model_name,
                    "backend": backend,
                    "branch": current_branch,
                    "active_dirs": active_dirs,
                    "session_history": session_history,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                print(f"\n{CYAN_MAIN}╭─────────────────── Session Backgrounded ─────────────────────╮{C_RESET}")
                print(f"  {CYAN_GLOW}Session ID:{C_RESET}      {WHITE_BOLD}{session_id}{C_RESET}")
                print(f"  {SLATE}Saved State:{C_RESET}     {WHITE_BOLD}{len(session_history)}{C_RESET} turns across branch {PURPLE_ACCENT}{current_branch}{C_RESET}")
                print(f"  {SLATE}Saved File:{C_RESET}      {SLATE_LIGHT}{saved_path}{C_RESET}")
                print(f"  {GREEN_ACCENT}Resume anytime:{C_RESET}  {WHITE_BOLD}python -m lunaite --resume {session_id}{C_RESET}")
                print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────────╯{C_RESET}\n")
                break

            # ─────────────────────────────────────────────────────────────
            # 6. /branch — Create conversation branch
            # ─────────────────────────────────────────────────────────────
            elif cmd == "/branch":
                if not arg:
                    print(f"  {PURPLE_ACCENT}Current Active Branch:{C_RESET} {WHITE_BOLD}{current_branch}{C_RESET}")
                    print(f"  {SLATE}Create or switch: {CYAN_MAIN}/branch <new_branch_name>{C_RESET}\n")
                else:
                    new_branch = arg.replace(" ", "-")
                    # Save checkpoint of current branch
                    save_session_to_disk({
                        "session_id": f"{session_id}_{current_branch}",
                        "model_name": model_name,
                        "backend": backend,
                        "branch": current_branch,
                        "active_dirs": active_dirs,
                        "session_history": list(session_history),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    current_branch = new_branch
                    print(f"\n  {GREEN_ACCENT}✓ Switched to branch:{C_RESET} {PURPLE_ACCENT}{C_BOLD}{current_branch}{C_RESET}")
                    print(f"  {SLATE}Forked {len(session_history)} turns. Divergent questions will now track on this branch.{C_RESET}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 7. /btw — Ask a quick side question without polluting history
            # ─────────────────────────────────────────────────────────────
            elif cmd == "/btw":
                if not arg:
                    print(f"  {SLATE}Usage: {CYAN_MAIN}/btw <quick question>{SLATE} to ask without polluting main conversation.{C_RESET}\n")
                    continue

                print(f"\n  {AMBER_ACCENT}┌─ {C_BOLD}BTW Side Question{C_RESET} {SLATE_DARK}(Isolated, not recorded in session history){C_RESET}")
                print(f"  {AMBER_ACCENT}└─{C_RESET} {WHITE_BOLD}{arg}{C_RESET}\n")

                btw_prompt = (
                    f"You are Lunaite AI. Answer this quick side question directly, accurately, and concisely:\n"
                    f"Question: {arg}\n\nAnswer:"
                )
                for chunk in model._raw_stream_generate(btw_prompt):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                print("\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 8. /bug — Export bug report and conversation transcript
            # ─────────────────────────────────────────────────────────────
            elif cmd == "/bug":
                stats = get_system_telemetry()
                bug_id = time.strftime("bug_%Y%m%d_%H%M%S")
                bug_file = Path.cwd() / f"lunaite_{bug_id}.json"

                report_data = {
                    "bug_id": bug_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "user_notes": arg or "No description provided",
                    "model": model_name,
                    "backend": backend,
                    "branch": current_branch,
                    "active_workspaces": active_dirs,
                    "system_telemetry": stats,
                    "recent_transcript": session_history[-10:],
                    "memory_insights": model.memory.data.get("episodic_insights", []) if model.memory else []
                }

                with open(bug_file, "w", encoding="utf-8") as bf:
                    json.dump(report_data, bf, indent=2)

                print(f"\n{CYAN_MAIN}╭────────────────── Bug Report Generated ──────────────────────╮{C_RESET}")
                print(f"  {GREEN_ACCENT}✓ Report Bundle:{C_RESET} {WHITE_BOLD}{bug_file.name}{C_RESET}")
                print(f"  {SLATE}Saved to:{C_RESET}        {SLATE_LIGHT}{bug_file}{C_RESET}")
                print(f"  {SLATE}Captured:{C_RESET}        System telemetry, environment, and last {len(session_history[-10:])} turns.")
                print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────────╯{C_RESET}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 9. /clear — Reset context; saves previous session to disk
            # ─────────────────────────────────────────────────────────────
            elif cmd in ["/clear", "/reset"]:
                if session_history:
                    save_session_to_disk({
                        "session_id": session_id,
                        "model_name": model_name,
                        "backend": backend,
                        "branch": current_branch,
                        "active_dirs": active_dirs,
                        "session_history": session_history,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    print(f"  {SLATE_DARK}Previous session saved ({session_id}). Resumable via /resume {session_id}{C_RESET}")

                session_id = time.strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:6]
                session_history.clear()
                model.clear_history()
                if model.memory:
                    model.memory.clear()
                print(f"  {GREEN_ACCENT}✓ Context cleared. Fresh session started.{C_RESET}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 10. /context — Visualize context usage as a colored grid
            # ─────────────────────────────────────────────────────────────
            elif cmd == "/context":
                mem_ctx = model.memory.get_context_summary() if model.memory else ""
                render_context_grid(
                    session_history=session_history,
                    model_name=model_name,
                    active_dirs=active_dirs,
                    current_branch=current_branch,
                    memory_ctx=mem_ctx,
                    max_context_tokens=max_context_tokens,
                    autocompact_pct=autocompact_pct
                )
                continue

            # ─────────────────────────────────────────────────────────────
            # 11. /compact — Free context by summarizing conversation
            # ─────────────────────────────────────────────────────────────
            elif cmd == "/compact":
                if not session_history:
                    print(f"  {SLATE}No active conversation to compact.{C_RESET}\n")
                    continue

                print(f"  {CYAN_MAIN}⟳ Compacting {len(session_history)} turns into high-density memory insights...{C_RESET}")
                
                # Build summary of history
                history_text = "\n".join([f"{r}: {t}" for r, t in session_history])
                compaction_prompt = (
                    f"Summarize the essential facts, decisions, code topics, and conclusions from this conversation "
                    f"into 3-4 dense bullet points for persistent memory injection:\n\n{history_text[:4000]}\n\nBullets:"
                )
                summary_bullets = model._raw_generate(compaction_prompt)

                if model.memory:
                    model.memory.add_insight(f"Compacted Session Summary: {summary_bullets[:300]}")

                # Keep only last turn in history
                tokens_freed = sum(len(txt.split()) for _, txt in session_history[:-2]) * 1.33 if len(session_history) > 2 else 0
                session_history = session_history[-2:]

                print(f"  {GREEN_ACCENT}✓ Context compacted! Freed ~{int(tokens_freed):,} tokens.{C_RESET}")
                print(f"  {SLATE}Persistent memory updated with conversation takeaways.{C_RESET}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 12. /resume — Resume saved sessions
            # ─────────────────────────────────────────────────────────────
            elif cmd == "/resume":
                saved_sessions = list_saved_sessions()
                if not arg:
                    if not saved_sessions:
                        print(f"  {SLATE}No saved sessions found in ~/.lunaite/sessions/{C_RESET}\n")
                    else:
                        print(f"\n{CYAN_MAIN}╭────────────────── Saved Sessions on Disk ───────────────────╮{C_RESET}")
                        for idx, s in enumerate(saved_sessions[:8], 1):
                            s_id = s.get("session_id", "unknown")
                            s_m = s.get("model_name", "unknown")
                            s_b = s.get("branch", "main")
                            turns = len(s.get("session_history", []))
                            time_str = s.get("updated_at", "recent")
                            print(f"  {CYAN_BRIGHT}[{idx}]{C_RESET} {WHITE_BOLD}{s_id:<28}{C_RESET} {SLATE_DARK}│{C_RESET} {PURPLE_ACCENT}{s_b:<8}{C_RESET} {SLATE_DARK}│{C_RESET} {turns:>2} turns {SLATE_DARK}│{C_RESET} {SLATE}{time_str}{C_RESET}")
                        print(f"\n  {SLATE}Usage: {CYAN_MAIN}/resume <session_id>{SLATE} to restore a session.{C_RESET}")
                        print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────────╯{C_RESET}\n")
                else:
                    loaded = load_session_from_disk(arg)
                    if loaded:
                        session_id = loaded.get("session_id", arg)
                        current_branch = loaded.get("branch", "main")
                        active_dirs = loaded.get("active_dirs", [os.getcwd()])
                        session_history = [(r, t) for r, t in loaded.get("session_history", [])]
                        print(f"  {GREEN_ACCENT}✓ Restored session '{session_id}' ({len(session_history)} turns, branch: {current_branch}).{C_RESET}\n")
                    else:
                        print(f"  {RED_ACCENT}Session '{arg}' not found.{C_RESET}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 13. /deliberate — Toggle cognitive reasoning
            # ─────────────────────────────────────────────────────────────
            elif cmd_lower == "/deliberate":
                deliberate = not deliberate
                mode_str = f"{GREEN_ACCENT}enabled{C_RESET}" if deliberate else f"{SLATE}disabled{C_RESET}"
                print(f"  {SLATE_DARK}Deliberation mode {mode_str}.{C_RESET}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 14. /history — View session conversation log
            # ─────────────────────────────────────────────────────────────
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

            # ─────────────────────────────────────────────────────────────
            # 15. /tools — Registered tool capabilities
            # ─────────────────────────────────────────────────────────────
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

            # ─────────────────────────────────────────────────────────────
            # 16. /info — Hardware diagnostics
            # ─────────────────────────────────────────────────────────────
            elif cmd_lower == "/info":
                stats = get_system_telemetry()
                print(f"\n{CYAN_MAIN}╭────────────────── System Diagnostics ────────────────────╮{C_RESET}")
                print(f"  {SLATE}CPU Utilization:{C_RESET} {WHITE_BOLD}{stats['cpu_percent']}%{C_RESET}   {SLATE}RAM:{C_RESET} {WHITE_BOLD}{stats['ram_used_gb']}/{stats['ram_total_gb']} GB{C_RESET} {SLATE_DARK}({stats['ram_percent']}%) {C_RESET}")
                print(f"  {SLATE}GPU Accelerator:{C_RESET} {WHITE_BOLD}{stats['gpu_name']}{C_RESET} {SLATE_DARK}({stats['gpu_vram_used_gb']} GB VRAM){C_RESET}")
                print(f"  {SLATE}Storage Free:{C_RESET}    {WHITE_BOLD}{stats['disk_free_gb']} GB available{C_RESET}")
                print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────╯{C_RESET}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 17. /models — List available models or switch model mid-session
            # ─────────────────────────────────────────────────────────────
            elif cmd_lower in ["/models", "/model"]:
                if not arg:
                    local_m, api_m = discover_available_models()
                    all_m = local_m + api_m
                    print(f"\n{CYAN_MAIN}╭──────────────────── Model Switcher ──────────────────────╮{C_RESET}")
                    print(f"  {SLATE}Active:{C_RESET} {GREEN_ACCENT}{C_BOLD}{model_name}{C_RESET} {SLATE_DARK}({backend or 'ollama'}){C_RESET}\n")
                    if local_m:
                        print(f"  {CYAN_BRIGHT}Local Models:{C_RESET}")
                        for idx, m in enumerate(local_m, 1):
                            marker = f"{GREEN_ACCENT}✓{C_RESET}" if m["name"] == model_name else " "
                            print(f"   {marker} {CYAN_MAIN}[{idx}]{C_RESET} {WHITE_BOLD}{m['name']:<24}{C_RESET} {SLATE_DARK}{m['desc']}{C_RESET}")
                    if api_m:
                        print(f"\n  {CYAN_BRIGHT}Cloud API Models:{C_RESET}")
                        for idx, m in enumerate(api_m, len(local_m) + 1):
                            marker = f"{GREEN_ACCENT}✓{C_RESET}" if m["name"] == model_name else " "
                            print(f"   {marker} {CYAN_MAIN}[{idx}]{C_RESET} {WHITE_BOLD}{m['name']:<24}{C_RESET} {SLATE_DARK}{m['desc']}{C_RESET}")
                    print(f"\n  {SLATE}Switch: {CYAN_MAIN}/model <name or number>{SLATE}  e.g. {CYAN_MAIN}/model qwen3:8b{C_RESET}")
                    print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────╯{C_RESET}\n")
                else:
                    local_m, api_m = discover_available_models()
                    all_m = local_m + api_m
                    new_model_name, new_backend = None, None

                    if arg.isdigit():
                        pick = int(arg) - 1
                        if 0 <= pick < len(all_m):
                            new_model_name = all_m[pick]["name"]
                            new_backend = all_m[pick]["backend"]
                    else:
                        for m in all_m:
                            if arg.lower() in m["name"].lower() or m["name"].lower() in arg.lower():
                                new_model_name = m["name"]
                                new_backend = m["backend"]
                                break
                        if not new_model_name:
                            new_model_name = arg
                            new_backend = "api" if arg.startswith(("gpt-", "claude-")) else "ollama"

                    if new_model_name:
                        old_name = model_name
                        model_name = new_model_name
                        backend = new_backend
                        model = wrap(model_name, backend=backend)
                        print(f"  {GREEN_ACCENT}✓ Switched:{C_RESET} {SLATE_DARK}{old_name}{C_RESET} → {WHITE_BOLD}{model_name}{C_RESET} {SLATE_DARK}({backend}){C_RESET}")
                        print(f"  {SLATE}History and memory are preserved across the switch.{C_RESET}\n")
                    else:
                        print(f"  {RED_ACCENT}Model not found:{C_RESET} {arg}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # 18. /help — Command overview
            # ─────────────────────────────────────────────────────────────
            elif cmd_lower in ["/help", "/?"]:
                print(f"\n{CYAN_MAIN}╭─────────────────── Available Commands ───────────────────╮{C_RESET}")
                print(f"  {CYAN_BRIGHT}/add-dir{C_RESET} [path]  {SLATE_LIGHT}Add new working directory to session context{C_RESET}")
                print(f"  {CYAN_BRIGHT}/cd{C_RESET} <path>       {SLATE_LIGHT}Move session to a new working directory{C_RESET}")
                print(f"  {CYAN_BRIGHT}/models{C_RESET} [name]   {SLATE_LIGHT}List available models or switch active model{C_RESET}")
                print(f"  {CYAN_BRIGHT}/context{C_RESET}         {SLATE_LIGHT}Visualize current context usage as a colored grid{C_RESET}")
                print(f"  {CYAN_BRIGHT}/compact{C_RESET}         {SLATE_LIGHT}Free context by summarizing conversation so far{C_RESET}")
                print(f"  {CYAN_BRIGHT}/autocompact{C_RESET}     {SLATE_LIGHT}Set context fill % before auto-summarizing{C_RESET}")
                print(f"  {CYAN_BRIGHT}/branch{C_RESET} [name]   {SLATE_LIGHT}Create a branch of current conversation{C_RESET}")
                print(f"  {CYAN_BRIGHT}/background{C_RESET}      {SLATE_LIGHT}Save session to disk & free the terminal{C_RESET}")
                print(f"  {CYAN_BRIGHT}/resume{C_RESET} [id]     {SLATE_LIGHT}List or resume saved background sessions{C_RESET}")
                print(f"  {CYAN_BRIGHT}/btw{C_RESET} <query>     {SLATE_LIGHT}Ask quick side question without polluting history{C_RESET}")
                print(f"  {CYAN_BRIGHT}/bug{C_RESET} [desc]      {SLATE_LIGHT}Export debug bundle & conversation transcript{C_RESET}")
                print(f"  {CYAN_BRIGHT}/clear{C_RESET}           {SLATE_LIGHT}Start fresh session (previous saved on disk){C_RESET}")
                print(f"  {CYAN_BRIGHT}/deliberate{C_RESET}     {SLATE_LIGHT}Toggle multi-perspective verification reasoning{C_RESET}")
                print(f"  {CYAN_BRIGHT}/history{C_RESET}        {SLATE_LIGHT}Review multi-turn conversation log{C_RESET}")
                print(f"  {CYAN_BRIGHT}/tools{C_RESET}          {SLATE_LIGHT}Inspect available tool registry & triggers{C_RESET}")
                print(f"  {CYAN_BRIGHT}/info{C_RESET}           {SLATE_LIGHT}Show hardware metrics & GPU telemetry{C_RESET}")
                print(f"  {CYAN_BRIGHT}/exit{C_RESET}           {SLATE_LIGHT}Gracefully save state & terminate session{C_RESET}")
                print(f"{CYAN_MAIN}╰──────────────────────────────────────────────────────────╯{C_RESET}\n")
                continue

            # ─────────────────────────────────────────────────────────────
            # Standard AI Generation & Tool Handling
            # ─────────────────────────────────────────────────────────────
            # Track user input in session history
            session_history.append(("You", user_input))

            # Auto-compact check
            history_words = sum(len(txt.split()) for _, txt in session_history)
            approx_tokens = int(history_words * 1.33) + 800
            if approx_tokens > (max_context_tokens * autocompact_pct / 100) and len(session_history) > 4:
                print(f"  {AMBER_ACCENT}⟳ Auto-compacting conversation history ({approx_tokens:,} tokens > {autocompact_pct}% limit)...{C_RESET}")
                summary_prompt = "Summarize previous discussion key points:\n" + "\n".join([f"{r}: {t[:100]}" for r, t in session_history[:-2]])
                summary_res = model._raw_generate(summary_prompt)
                if model.memory:
                    model.memory.add_insight(f"Auto-Compacted: {summary_res[:200]}")
                session_history = session_history[-2:]

            # Response header
            print(f"\n{CYAN_GLOW}{C_BOLD}✦ Lunaite{C_RESET}")

            # Execution with tool awareness and deliberation callbacks
            memory_ctx = model.memory.get_context_summary() if model.memory else ""
            enriched_ctx = f"Q: {last_web_topic}" if last_web_topic else memory_ctx
            intent = model.agent.decide_tool(user_input, lambda p: model._raw_generate(p), context=enriched_ctx) if model.agent else None

            if deliberate and model.cognitive:
                response = model.cognitive.deliberate(
                    user_input,
                    lambda p: model._raw_generate(p),
                    progress_callback=render_deliberation_status
                )
                print(f"\n{response}\n")
                session_history.append(("Lunaite", response))
            elif intent:
                render_tool_call(intent[0], intent[1])
                tool_output = model.agent.execute_tool(intent[0], intent[1])
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
                    f"Active Workspaces: {', '.join(active_dirs)}\n"
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
                    path_match = re.search(r'([A-Za-z]:\\[^\n\r"<>|*?]{3,})', tool_output)
                    if path_match:
                        fpath = path_match.group(1).strip()
                        if os.path.isfile(fpath):
                            ext = os.path.splitext(fpath)[1].lower()
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
                    ws_note = f"\nActive Workspaces: {', '.join(active_dirs)}"
                    full_prompt = f"{system_prompt}{ws_note}\n\nUser: {user_input}\nLunaite AI:" if system_prompt else user_input

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
                    session_history.append(("Lunaite", response))

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
    run_parser.add_argument("--resume", default=None, help="Resume saved session by ID or filename")

    # Info command
    subparsers.add_parser("info", help="Show system telemetry and diagnostics")

    # Direct flag support
    parser.add_argument("--resume", default=None, help="Resume saved session by ID or filename")
    parser.add_argument("--deliberate", action="store_true", help="Enable multi-perspective cognitive deliberation")
    parser.add_argument("--backend", default=None, help="Backend: ollama, api, or huggingface")

    args = parser.parse_args()

    resume_id = getattr(args, "resume", None)

    if args.command == "run" or args.command is None:
        model_name = getattr(args, "model", None)
        deliberate = getattr(args, "deliberate", False)
        backend = getattr(args, "backend", None)
        run_chat_cli(model_name=model_name, backend=backend, deliberate=deliberate, resume_id=resume_id)

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
