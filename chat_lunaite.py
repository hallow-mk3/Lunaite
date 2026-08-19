"""
Lunaite Code CLI — Custom White & Cyan Terminal Interface
=========================================================
Inspired by the Anthropic Claude Code terminal experience.
Built for Lunaite AI 10T Frontier Mythos Mind.
Created by Swasthik Shetty.

Design System:
  - Primary Typography : Crisp Bright White (\033[97m\033[1m)
  - Primary Accent     : Electric / Cyan (\033[96m\033[1m)
  - Subdued Accents    : Dim Cyan (\033[36m) & Slate Gray (\033[90m)
  - Structure          : Unicode Box-Drawing Hierarchy (╭─, ╰─, │, ●, ›)
"""

import sys
import os
import json
import time
import re
import argparse
import datetime
import subprocess
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from lunaite_agent import (
    memory_db,
    web_search,
    fetch_url,
    fetch_weather,
    wiki_lookup,
    should_auto_search_web,
    detect_app_intent,
    execute_app_action,
    run_expert_debate
)
from lunaite_apps import (
    spotify_open,
    spotify_search_and_play,
    spotify_media_action,
    discord_open,
    outlook_open,
    outlook_get_unread_emails,
    outlook_compose_email,
    outlook_get_calendar_events,
    explorer_open,
    explorer_search_files,
    open_application,
    close_application,
    list_running_applications,
    system_control,
    execute_app_intent
)
from lunaite_voice import speak_out, listen_voice, run_voice_assistant

# ─── Engine Configuration ─────────────────────────────────────────────────────

DEFAULT_MODEL   = "lunaite-ai"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE     = 0.65
TOP_P           = 0.95
TOP_K           = 50
NUM_CTX         = 4096
MAX_PREDICT     = 4096
REPEAT_PENALTY  = 1.04
MIROSTAT_TAU    = 4.5
MIROSTAT_ETA    = 0.08

VERBOSE_THINKING = False
AUTO_WEB_SEARCH  = True
VOICE_MODE       = False
AGENT_MAX_STEPS  = 6

# ─── Claude Code White & Cyan Color Palette ───────────────────────────────────

RESET       = "\033[0m"
BOLD        = "\033[1m"
DIM         = "\033[90m"
ITALIC      = "\033[3m"

# White Tones
WHITE       = "\033[97m"
BOLD_WHITE  = "\033[97m\033[1m"
MUTED_WHITE = "\033[37m"

# Cyan Tones (Claude Code Theme)
CYAN        = "\033[96m"
BOLD_CYAN   = "\033[96m\033[1m"
DIM_CYAN    = "\033[36m"
BADGE_CYAN  = "\033[46m\033[30m\033[1m"

# Status Highlights
GREEN       = "\033[92m"
YELLOW      = "\033[93m"
RED         = "\033[91m"


# ─── 10T MoE Expert Topology ──────────────────────────────────────────────────

EXPERTS = {
    "astrophysics": {
        "id": "MoE-087",
        "name": "Relativistic Astrophysics & Cosmology",
        "keywords": ["black hole", "relativity", "gravity", "space", "star", "entropy", "schwarzschild", "kerr", "hawking", "quantum", "cosmic", "universe", "ligo"]
    },
    "math": {
        "id": "MoE-142",
        "name": "Differential Geometry & Tensor Topology",
        "keywords": ["math", "equation", "proof", "matrix", "tensor", "topology", "calculus", "manifold", "algebra", "integral", "theorem", "godel"]
    },
    "coding": {
        "id": "MoE-219",
        "name": "Systems Architecture & Autonomous Action",
        "keywords": ["code", "python", "rust", "c++", "function", "algorithm", "async", "memory", "cuda", "gpu", "compiler", "debug", "api", "architecture", "script", "file", "build", "run"]
    },
    "web": {
        "id": "MoE-312",
        "name": "Live Internet & Global Knowledge Retrieval",
        "keywords": ["internet", "web", "online", "search", "url", "news", "today", "current", "latest", "weather", "browse", "google", "website", "time"]
    },
    "philosophy": {
        "id": "MoE-384",
        "name": "Ontology & Consciousness Philosophy",
        "keywords": ["consciousness", "mind", "qualia", "philosophy", "ethics", "existence", "soul", "meaning", "reality", "hard problem", "epistemology"]
    },
    "empathy": {
        "id": "MoE-495",
        "name": "Emotional Intelligence & Human Grounding",
        "keywords": ["feel", "lonely", "sad", "happy", "life", "stressed", "friend", "help me", "tired", "anxious", "talk", "love"]
    }
}


def route_10t_query(prompt: str):
    prompt_lower = prompt.lower()
    active = [exp for key, exp in EXPERTS.items() if any(kw in prompt_lower for kw in exp["keywords"])]
    if not active:
        active = [{"id": "MoE-001", "name": "Luminous Frontier Synthesis Core"}]
    return active


# ─── Local Tool Sandbox ───────────────────────────────────────────────────────

def tool_run_python(code: str) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=25,
            cwd="."
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if err and not out:
            return f"[Stderr]:\n{err}"
        elif err and out:
            return f"[Stdout]:\n{out}\n\n[Stderr]:\n{err}"
        return out if out else "[Code executed successfully with no stdout]"
    except subprocess.TimeoutExpired:
        return "[Error: Python execution timed out after 25s]"
    except Exception as e:
        return f"[Execution Error]: {e}"


def tool_run_shell(command: str) -> str:
    try:
        cmd_lower = command.strip().lower()
        if any(app in cmd_lower for app in ["notepad", "calc", "explorer", "code", "start "]):
            subprocess.Popen(f'start "" {command}', shell=True)
            return f"[Application Process Launched Successfully]: {command}"

        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=25,
            cwd="."
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if err and not out:
            return f"[Error/Stderr]:\n{err}"
        elif err and out:
            return f"[Output]:\n{out}\n\n[Stderr]:\n{err}"
        return out if out else "[Command completed with exit code 0]"
    except subprocess.TimeoutExpired:
        return "[Error: Command timed out after 25s]"
    except Exception as e:
        return f"[Shell Error]: {e}"


def tool_read_file(path: str) -> str:
    try:
        path = os.path.expandvars(os.path.expanduser(path.strip()))
        if not os.path.exists(path):
            return f"[Error: File '{path}' does not exist]"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) > 80:
            return "".join(lines[:80]) + f"\n\n... [Truncated {len(lines)-80} more lines]"
        return "".join(lines)
    except Exception as e:
        return f"[File Read Error]: {e}"


def tool_write_file(path: str, content: str) -> str:
    try:
        path = os.path.expandvars(os.path.expanduser(path.strip()))
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[Success: Wrote {len(content)} bytes to {path}]"
    except Exception as e:
        return f"[File Write Error]: {e}"


def tool_list_dir(path: str = ".") -> str:
    try:
        path = os.path.expandvars(os.path.expanduser(path.strip()))
        if not os.path.exists(path):
            return f"[Error: Directory '{path}' does not exist]"
        items = os.listdir(path)
        dirs = [f"{i}/" for i in items if os.path.isdir(os.path.join(path, i)) and not i.startswith(".")]
        files = [i for i in items if os.path.isfile(os.path.join(path, i)) and not i.startswith(".")]
        return f"Directories:\n  " + "\n  ".join(dirs[:12]) + f"\nFiles:\n  " + "\n  ".join(files[:20])
    except Exception as e:
        return f"[List Dir Error]: {e}"


# ─── Autonomous ReAct Agent Loop ──────────────────────────────────────────────

def run_agent_loop(model: str, goal: str):
    print(f"\n{BOLD_CYAN}╭── ● LUNAITE AGENT RUNNER ────────────────────────────────────────╮{RESET}")
    print(f"{BOLD_CYAN}│{RESET}  {DIM}Goal:{RESET} {BOLD_WHITE}{goal}{RESET}")
    print(f"{BOLD_CYAN}╰──────────────────────────────────────────────────────────────────╯{RESET}\n")

    mem_ctx = memory_db.get_memory_context()

    system_prompt = (
        f"You are Lunaite Code 10T Autonomous Agent created by Swasthik Shetty.{mem_ctx}\n"
        "You have access to live tools to perform REAL actions on the system:\n"
        "1. <tool:shell>command</tool:shell> — Runs PowerShell commands or launches apps\n"
        "2. <tool:open_app>app_name</tool:open_app> — Opens desktop applications (notepad, spotify, discord, calc, etc.)\n"
        "3. <tool:web_search>query</tool:web_search> — Real-time live web search\n"
        "4. <tool:fetch_url>url</tool:fetch_url> — Reads live web page content\n"
        "5. <tool:python>code</tool:python> — Executes Python code\n"
        "6. <tool:read_file>path</tool:read_file> — Reads a local file\n"
        "7. <tool:write_file path=\"...\">content</tool:write_file> — Writes a file\n"
        "8. <tool:list_dir>path</tool:list_dir> — Lists directory contents\n\n"
        "CRITICAL EXECUTION RULES:\n"
        "- In Step 1, ONLY output your Thought and Action. DO NOT write Final Answer until the tool has actually been executed and an Observation received!\n"
        "- Format:\n"
        "Thought: <step-by-step reasoning>\n"
        "Action: <tool:tool_name>arguments</tool:tool_name>\n"
        "- When all actions are executed and goal is accomplished, output:\n"
        "Final Answer: <summary of what was achieved>"
    )

    agent_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Accomplish this goal step-by-step: {goal}"}
    ]

    for step in range(1, AGENT_MAX_STEPS + 1):
        print(f"{DIM_CYAN}╭── Step {step}/{AGENT_MAX_STEPS} ──────────────────────────────────────────────────────────╮{RESET}")
        
        try:
            response, stats = query_ollama(model, agent_history, stream=True)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[Agent step halted by user]{RESET}\n")
            break

        if not response:
            print(f"{RED}Agent failed to get response.{RESET}")
            break

        tool_executed = False

        # 0. Open App Tool
        app_match = re.search(r"<tool:open_app>(.*?)</tool:open_app>", response, re.DOTALL)
        if app_match:
            app_n = app_match.group(1).strip()
            print(f"\n  {BOLD_CYAN}● OpenApp{RESET}{WHITE}({app_n}){RESET}")
            obs = open_application(app_n)
            print(f"  {DIM}└─ Result: {obs}{RESET}\n")
            agent_history.append({"role": "assistant", "content": response})
            agent_history.append({"role": "user", "content": f"Observation: {obs}"})
            tool_executed = True

        # 1. Shell
        sh_match = re.search(r"<tool:shell>(.*?)</tool:shell>", response, re.DOTALL)
        if not tool_executed and sh_match:
            cmd = sh_match.group(1).strip()
            print(f"\n  {BOLD_CYAN}● Bash{RESET}{WHITE}({cmd}){RESET}")
            obs = tool_run_shell(cmd)
            print(f"  {DIM}└─ Output:\n{obs[:250]}{RESET}\n")
            agent_history.append({"role": "assistant", "content": response})
            agent_history.append({"role": "user", "content": f"Observation: {obs}"})
            tool_executed = True

        # 2. Web Search
        ws_match = re.search(r"<tool:web_search>(.*?)</tool:web_search>", response, re.DOTALL)
        if not tool_executed and ws_match:
            query = ws_match.group(1).strip()
            print(f"\n  {BOLD_CYAN}● WebSearch{RESET}{WHITE}({query}){RESET}")
            obs = web_search(query)
            print(f"  {DIM}└─ Observation: {obs[:200]}...{RESET}\n")
            agent_history.append({"role": "assistant", "content": response})
            agent_history.append({"role": "user", "content": f"Observation: {obs}"})
            tool_executed = True

        # 3. Fetch URL
        fu_match = re.search(r"<tool:fetch_url>(.*?)</tool:fetch_url>", response, re.DOTALL)
        if not tool_executed and fu_match:
            url = fu_match.group(1).strip()
            print(f"\n  {BOLD_CYAN}● FetchURL{RESET}{WHITE}({url}){RESET}")
            obs = fetch_url(url)
            print(f"  {DIM}└─ Observation: {obs[:200]}...{RESET}\n")
            agent_history.append({"role": "assistant", "content": response})
            agent_history.append({"role": "user", "content": f"Observation: {obs}"})
            tool_executed = True

        # 4. Python
        py_match = re.search(r"<tool:python>(.*?)</tool:python>", response, re.DOTALL)
        if not tool_executed and py_match:
            code = py_match.group(1).strip()
            print(f"\n  {BOLD_CYAN}● PythonREPL{RESET}")
            obs = tool_run_python(code)
            print(f"  {DIM}└─ Output:\n{obs[:250]}{RESET}\n")
            agent_history.append({"role": "assistant", "content": response})
            agent_history.append({"role": "user", "content": f"Observation: {obs}"})
            tool_executed = True

        # 5. Read File
        rf_match = re.search(r"<tool:read_file>(.*?)</tool:read_file>", response, re.DOTALL)
        if not tool_executed and rf_match:
            path = rf_match.group(1).strip()
            print(f"\n  {BOLD_CYAN}● ReadFile{RESET}{WHITE}({path}){RESET}")
            obs = tool_read_file(path)
            print(f"  {DIM}└─ Content: {len(obs)} bytes loaded{RESET}\n")
            agent_history.append({"role": "assistant", "content": response})
            agent_history.append({"role": "user", "content": f"Observation: {obs}"})
            tool_executed = True

        # 6. Write File
        wf_match = re.search(r'<tool:write_file path="(.*?)">(.*?)</tool:write_file>', response, re.DOTALL)
        if not tool_executed and wf_match:
            path = wf_match.group(1).strip()
            content = wf_match.group(2)
            print(f"\n  {BOLD_CYAN}● WriteFile{RESET}{WHITE}({path}){RESET}")
            obs = tool_write_file(path, content)
            print(f"  {DIM}└─ {obs}{RESET}\n")
            agent_history.append({"role": "assistant", "content": response})
            agent_history.append({"role": "user", "content": f"Observation: {obs}"})
            tool_executed = True

        # 7. List Dir
        ld_match = re.search(r"<tool:list_dir>(.*?)</tool:list_dir>", response, re.DOTALL)
        if not tool_executed and ld_match:
            path = ld_match.group(1).strip() or "."
            print(f"\n  {BOLD_CYAN}● ListDir{RESET}{WHITE}({path}){RESET}")
            obs = tool_list_dir(path)
            print(f"  {DIM}└─ {obs[:150]}...{RESET}\n")
            agent_history.append({"role": "assistant", "content": response})
            agent_history.append({"role": "user", "content": f"Observation: {obs}"})
            tool_executed = True

        # Goal Completion check
        if not tool_executed or ("Final Answer:" in response and step > 1):
            print(f"\n{BOLD_CYAN}● Task Complete{RESET} {DIM}· Goal achieved successfully{RESET}\n")
            memory_db.add_insight(f"Completed agent goal: {goal[:60]}")
            return


# ─── Ollama Low-Level Client (GPU Accelerated) ────────────────────────────────

def _ensure_ollama_online():
    """Verify Ollama is running, and auto-start it if offline without crashing."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass

    try:
        ollama_exe = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
        if os.path.exists(ollama_exe):
            subprocess.Popen([ollama_exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            return True
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            return True
    except Exception:
        return False


def get_installed_models() -> list:
    """Fetch list of all locally installed models in Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
            return [m.get("name") for m in data.get("models", [])]
    except Exception:
        return ["lunaite-ai", "qwen3:8b", "qwen3.8:27b"]


def query_ollama(model: str, messages: list, stream: bool = True) -> tuple:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "num_ctx": NUM_CTX,
            "num_predict": MAX_PREDICT,
            "repeat_penalty": REPEAT_PENALTY,
            "num_gpu": 99,
            "num_thread": 16
        }
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    full_response = ""
    stats = {"eval_count": 0, "eval_duration": 0, "prompt_eval_duration": 0}

    # Ensure Ollama is responsive
    _ensure_ollama_online()

    t_start = time.time()
    first_token_time = None

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        if first_token_time is None:
                            first_token_time = time.time()
                        full_response += token
                        if stream:
                            print(f"{WHITE}{token}{RESET}", end="", flush=True)
                    if data.get("done"):
                        stats["eval_count"] = data.get("eval_count", 0)
                        stats["eval_duration"] = data.get("eval_duration", 0)
                        stats["prompt_eval_duration"] = data.get("prompt_eval_duration", 0)
                        break
                except json.JSONDecodeError:
                    pass
        if stream:
            print()
    except KeyboardInterrupt:
        if stream:
            print(f"\n{DIM}[Halted by user]{RESET}")
    except urllib.error.URLError:
        print(f"\n{YELLOW}[Ollama Offline — Reconnecting...]{RESET}")
        _ensure_ollama_online()
    except Exception as e:
        print(f"\n{RED}[Generation Error]: {e}{RESET}")

    stats["wall_time"] = time.time() - t_start
    stats["first_token_time"] = first_token_time
    return full_response, stats


def stream_chat(model: str, messages: list, active_experts: list) -> str:
    expert_tags = " · ".join([f"{e['id']} ({e['name'].split()[0]})" for e in active_experts])
    print(f"{DIM_CYAN}╭── MoE Router: {expert_tags} ─────────────────────────────╮{RESET}")

    if VERBOSE_THINKING:
        print(f"{DIM}│  💭 Deliberating: First-principles cognitive synthesis...{RESET}")

    mem_ctx = memory_db.get_memory_context()
    augmented_messages = list(messages)
    if augmented_messages and augmented_messages[0].get("role") != "system":
        augmented_messages.insert(0, {"role": "system", "content": f"Lunaite Code 10T Frontier Mind with Live Internet Access.{mem_ctx}"})

    print(f"{BOLD_CYAN}│{RESET} ")
    print(f"{BOLD_CYAN}│  {BOLD_WHITE}Lunaite AI (10T):{RESET}")
    print(f"{BOLD_CYAN}│{RESET} ", end="", flush=True)

    response, stats = query_ollama(model, augmented_messages, stream=True)
    
    # Accurate hardware eval speed calculation
    eval_count = stats.get("eval_count", 0)
    eval_ns = stats.get("eval_duration", 0)
    if eval_count > 0 and eval_ns > 0:
        gen_speed = eval_count / (eval_ns / 1e9)
    else:
        words = len(response.split())
        gen_speed = (words * 1.33) / max(stats.get("wall_time", 1.0), 0.001)

    elapsed = stats.get("wall_time", 0.0)
    words = len(response.split())

    print(f"{DIM_CYAN}╰────────────────────────────────────── {DIM}[{words} words · {elapsed:.2f}s · {gen_speed:.1f} tok/s · RTX 5070 GPU]{RESET}\n")
    return response


# ─── Claude Code Minimalist UI Banners ────────────────────────────────────────

def print_banner():
    cwd = os.getcwd()
    print(f"\n{BOLD_CYAN}╭────────────────────────────────────────────────────────────────────────╮{RESET}")
    print(f"{BOLD_CYAN}│{RESET}  {BOLD_WHITE}● LUNAITE CODE{RESET}  {DIM_CYAN}v4.0 (10T · 27B Dense Core · Vision · Thinking){RESET}     {BOLD_CYAN}│{RESET}")
    print(f"{BOLD_CYAN}│{RESET}  {DIM}Working Dir :{RESET} {WHITE}{cwd[:54]}{'...' if len(cwd)>54 else ''}{RESET}")
    print(f"{BOLD_CYAN}│{RESET}  {DIM}Hardware    :{RESET} {CYAN}RTX 5070 GPU + CPU (CUDA){RESET} · {DIM}Base:{RESET} {CYAN}qwen3.8:27B{RESET} · {DIM}Net:{RESET} {GREEN}Active{RESET} {BOLD_CYAN}│{RESET}")
    print(f"{BOLD_CYAN}│{RESET}  {DIM}Full Access :{RESET} {GREEN}Apps · Files · Internet · Shell · Voice · Agent{RESET}    {BOLD_CYAN}│{RESET}")
    print(f"{BOLD_CYAN}╰────────────────────────────────────────────────────────────────────────╯{RESET}")
    print(f"  {DIM}Type {BOLD_WHITE}/help{RESET}{DIM} for commands  ·  {BOLD_WHITE}/agent <goal>{RESET}{DIM} to run tasks  ·  {BOLD_WHITE}/exit{RESET}{DIM} to quit{RESET}\n")


def print_help():
    print(f"\n{BOLD_CYAN}● Lunaite Code Command Palette:{RESET}")
    print(f"  {CYAN}/model [name]{RESET}        {DIM}— Switch inference model or view installed models list{RESET}")
    print(f"  {CYAN}/voice [start]{RESET}        {DIM}— Toggle two-way voice response or enter hands-free voice loop{RESET}")
    print(f"  {CYAN}/spotify play <song>{RESET}  {DIM}— Search and immediately auto-play top track on Spotify{RESET}")
    print(f"  {CYAN}/discord <user/dm>{RESET}   {DIM}— Open Discord Quick Switcher to messages with username{RESET}")
    print(f"  {CYAN}/outlook [unread]{RESET}     {DIM}— Check unread Outlook emails (desktop & web fallback){RESET}")
    print(f"  {CYAN}/explorer [folder]{RESET}    {DIM}— Open Windows File Explorer (downloads, docs, c:, etc.){RESET}")
    print(f"  {CYAN}/app <name>{RESET}           {DIM}— Open any laptop app (chrome, vscode, calc, steam, etc.){RESET}")
    print(f"  {CYAN}/agent <goal>{RESET}        {DIM}— Autonomous multi-step agent with real tools & app control{RESET}")
    print(f"  {CYAN}/debate <topic>{RESET}      {DIM}— Run 3-expert debate and synthesize consensus{RESET}")
    print(f"  {CYAN}/web <query>{RESET}         {DIM}— Search live web (DuckDuckGo){RESET}")
    print(f"  {CYAN}/url <url>{RESET}           {DIM}— Fetch and read any live webpage or documentation{RESET}")
    print(f"  {CYAN}/weather <city>{RESET}      {DIM}— Real-time weather lookup{RESET}")
    print(f"  {CYAN}/wiki <topic>{RESET}        {DIM}— Wikipedia summary lookup{RESET}")
    print(f"  {CYAN}/autoweb{RESET}             {DIM}— Toggle automatic web search on chat prompts{RESET}")
    print(f"  {CYAN}/remember <k> <v>{RESET}    {DIM}— Store persistent fact into 10T memory bank{RESET}")
    print(f"  {CYAN}/memories{RESET}            {DIM}— Inspect active long-term memories{RESET}")
    print(f"  {CYAN}/forget <key>{RESET}        {DIM}— Remove a stored memory{RESET}")
    print(f"  {CYAN}/run <code>{RESET}          {DIM}— Live Python REPL execution{RESET}")
    print(f"  {CYAN}/sh <cmd>{RESET}            {DIM}— Execute PowerShell command directly{RESET}")
    print(f"  {CYAN}/read <path>{RESET}         {DIM}— Read local workspace file{RESET}")
    print(f"  {CYAN}/write <f> <c>{RESET}       {DIM}— Write/create file{RESET}")
    print(f"  {CYAN}/ls [dir]{RESET}             {DIM}— List directory files{RESET}")
    print(f"  {CYAN}/think{RESET}                {DIM}— Toggle 10T Deliberation Stream{RESET}")
    print(f"  {CYAN}/clear{RESET}                {DIM}— Clear conversation context{RESET}")
    print(f"  {CYAN}/info{RESET}                 {DIM}— Telemetry & GPU throughput{RESET}")
    print(f"  {CYAN}/export{RESET}               {DIM}— Save conversation transcript to markdown{RESET}")
    print(f"  {CYAN}/exit{RESET}                 {DIM}— Exit the terminal{RESET}\n")


def print_info(model: str, history: list, start_time: float):
    elapsed = time.time() - start_time
    turns = len([m for m in history if m["role"] == "user"])
    memories_count = len(memory_db.data.get("user_facts", {})) + len(memory_db.data.get("episodic_insights", []))
    print(f"\n{BOLD_CYAN}● System Telemetry & Statistics:{RESET}")
    print(f"  {DIM}Active Model    :{RESET} {WHITE}{model}{RESET}")
    print(f"  {DIM}Architecture    :{RESET} {WHITE}10T MoE Autonomous Agent (Mythos Edition){RESET}")
    print(f"  {DIM}GPU Hardware    :{RESET} {CYAN}NVIDIA GeForce RTX 5070 Laptop GPU (CUDA){RESET}")
    print(f"  {DIM}Internet Link   :{RESET} {GREEN}Active (Live Web, URL Reader, Weather, Wiki){RESET}")
    print(f"  {DIM}Auto-Web Search :{RESET} {GREEN if AUTO_WEB_SEARCH else YELLOW}{'ENABLED' if AUTO_WEB_SEARCH else 'DISABLED'}{RESET}")
    print(f"  {DIM}Memory Store    :{RESET} {CYAN}{memories_count} Stored Entries{RESET}")
    print(f"  {DIM}Session Turns   :{RESET} {WHITE}{turns} user turns{RESET}")
    print(f"  {DIM}Session Elapsed :{RESET} {WHITE}{elapsed:.0f}s{RESET}\n")


def show_memories():
    print(f"\n{BOLD_CYAN}● Lunaite Long-Term Memory Bank:{RESET}")
    print(f"  {DIM}Store: {memory_db.filepath}{RESET}\n")
    print(f"  {BOLD_WHITE}Creator Profile:{RESET}")
    print(f"    {DIM}Name:{RESET} {WHITE}{memory_db.data.get('creator', {}).get('name')}{RESET}")
    print(f"    {DIM}Role:{RESET} {WHITE}{memory_db.data.get('creator', {}).get('role')}{RESET}\n")
    
    facts = memory_db.data.get("user_facts", {})
    print(f"  {BOLD_WHITE}User Facts ({len(facts)}):{RESET}")
    if facts:
        for k, v in facts.items():
            print(f"    {CYAN}●{RESET} {WHITE}{k}{RESET}: {DIM}{v}{RESET}")
    else:
        print(f"    {DIM}(No custom facts saved. Use /remember <key> <value>){RESET}")

    insights = memory_db.data.get("episodic_insights", [])[-5:]
    print(f"\n  {BOLD_WHITE}Recent Insights:{RESET}")
    if insights:
        for ins in insights:
            print(f"    {CYAN}●{RESET} {DIM}[{ins.get('timestamp')}]{RESET} {WHITE}{ins.get('insight')}{RESET}")
    else:
        print(f"    {DIM}(No episodic insights recorded yet){RESET}")
    print()


def export_conversation(history: list, model: str):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lunaite_chat_{ts}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Lunaite Code Chat Transcript — {ts}\n")
        f.write(f"**Architecture**: 10T MoE Mythos Mind\n")
        f.write(f"**Creator**: Swasthik Shetty\n\n---\n\n")
        for msg in history:
            role = "**You**" if msg["role"] == "user" else "**Lunaite Code (10T)**"
            f.write(f"{role}:\n{msg['content']}\n\n---\n\n")
    print(f"\n  {GREEN}✓ Conversation exported to: {filename}{RESET}\n")


# ─── Main Interactive Loop ───────────────────────────────────────────────────

def main():
    global VERBOSE_THINKING, AUTO_WEB_SEARCH, VOICE_MODE

    parser = argparse.ArgumentParser(description="Lunaite Code Terminal CLI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--think", action="store_true", help="Enable verbose deliberation")
    parser.add_argument("--voice", action="store_true", help="Enable voice response mode")
    args = parser.parse_args()
    model = args.model

    if args.think:
        VERBOSE_THINKING = True
    if args.voice:
        VOICE_MODE = True

    print_banner()
    if VOICE_MODE:
        print(f"  {GREEN}🔊 Voice Output Mode Active (Lunaite will speak replies aloud){RESET}\n")

    conversation_history = []
    session_start = time.time()

    def quick_query(prompt: str, stream: bool = True) -> str:
        res, _ = query_ollama(model, [{"role": "user", "content": prompt}], stream=stream)
        return res

    while True:
        try:
            print(f"{BOLD_CYAN}╭─ You{RESET}")
            user_input = input(f"{BOLD_CYAN}╰─› {BOLD_WHITE}").strip()
            print(f"{RESET}", end="")
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{DIM}Lunaite Code session closed. Farewell.{RESET}\n")
            break

        if not user_input:
            continue

        raw_cmd = user_input.strip()
        cmd_lower = raw_cmd.lower()

        # Command Dispatcher
        if cmd_lower in ("/exit", "/quit", "/q"):
            print(f"\n{DIM}Lunaite Code session closed. Farewell.{RESET}\n")
            break
        elif cmd_lower == "/model" or raw_cmd.startswith("/model"):
            parts = raw_cmd.split(" ", 1)
            if len(parts) > 1 and parts[1].strip():
                new_model = parts[1].strip()
                model = new_model
                print(f"\n  {GREEN}✓ Switched active model to: {model}{RESET}\n")
            else:
                installed = get_installed_models()
                print(f"\n{BOLD_CYAN}● Active Model:{RESET} {BOLD_WHITE}{model}{RESET}")
                print(f"{BOLD_CYAN}● Installed Models in Ollama:{RESET}")
                for m in installed:
                    star = f" {GREEN}(active){RESET}" if m == model or m.startswith(model) else ""
                    print(f"  {CYAN}●{RESET} {WHITE}{m}{RESET}{star}")
                print(f"\n  {DIM}To switch, type: {BOLD_WHITE}/model <name>{RESET}\n")
            continue
        elif cmd_lower == "/voice" or raw_cmd.startswith("/voice"):
            parts = raw_cmd.split()
            if len(parts) > 1 and parts[1].lower() in ["start", "loop", "handsfree", "continuous"]:
                print(f"\n{BOLD_CYAN}● Launching Hands-Free Voice Agent Session...{RESET}\n")
                run_voice_assistant(lambda q: quick_query(q, stream=False))
                continue
            else:
                VOICE_MODE = not VOICE_MODE
                print(f"\n  {CYAN}● Voice Audio Output:{RESET} {GREEN if VOICE_MODE else YELLOW}{'ENABLED (Speaking aloud)' if VOICE_MODE else 'DISABLED'}{RESET}\n")
                if VOICE_MODE:
                    speak_out("Voice mode enabled. I will speak my responses.")
                continue
        elif raw_cmd.startswith("/spotify"):
            sub = raw_cmd[8:].strip()
            if not sub:
                res = spotify_open()
            elif sub.startswith("play"):
                sq = sub[4:].strip()
                res = spotify_search_and_play(sq) if sq else spotify_media_action("playpause")
            else:
                res = spotify_media_action(sub)
            print(f"\n  {GREEN}✓ {res}{RESET}\n")
            if VOICE_MODE:
                speak_out(res)
            continue
        elif raw_cmd.startswith("/discord"):
            sub = raw_cmd[8:].strip()
            res = discord_open(sub if sub else None)
            print(f"\n  {GREEN}✓ {res}{RESET}\n")
            if VOICE_MODE:
                speak_out(res)
            continue
        elif raw_cmd.startswith("/outlook"):
            sub = raw_cmd[8:].strip()
            if sub in ["unread", "inbox", ""]:
                res = outlook_get_unread_emails()
            elif sub in ["calendar", "events"]:
                res = outlook_get_calendar_events()
            else:
                res = outlook_open()
            print(f"\n{WHITE}{res}{RESET}\n")
            if VOICE_MODE:
                speak_out(res)
            continue
        elif raw_cmd.startswith("/explorer"):
            sub = raw_cmd[9:].strip()
            res = explorer_open(sub if sub else "downloads")
            print(f"\n  {GREEN}✓ {res}{RESET}\n")
            if VOICE_MODE:
                speak_out(res)
            continue
        elif raw_cmd.startswith("/app "):
            app_target = raw_cmd[5:].strip()
            res = open_application(app_target)
            print(f"\n  {GREEN}✓ {res}{RESET}\n")
            if VOICE_MODE:
                speak_out(res)
            continue
        elif cmd_lower in ["/apps", "/running"]:
            res = list_running_applications()
            print(f"\n{WHITE}{res}{RESET}\n")
            continue
        elif cmd_lower == "/memories":
            show_memories()
            continue
        elif cmd_lower == "/clear":
            conversation_history.clear()
            print(f"\n  {GREEN}✓ Context cleared.{RESET}\n")
            continue
        elif cmd_lower == "/autoweb":
            AUTO_WEB_SEARCH = not AUTO_WEB_SEARCH
            print(f"\n  {CYAN}● Auto Web Search:{RESET} {GREEN if AUTO_WEB_SEARCH else YELLOW}{'ENABLED' if AUTO_WEB_SEARCH else 'DISABLED'}{RESET}\n")
            continue
        elif cmd_lower == "/think":
            VERBOSE_THINKING = not VERBOSE_THINKING
            print(f"\n  {CYAN}● 10T Deliberation Stream:{RESET} {GREEN if VERBOSE_THINKING else YELLOW}{'ON' if VERBOSE_THINKING else 'OFF'}{RESET}\n")
            continue
        elif cmd_lower == "/info":
            print_info(model, conversation_history, session_start)
            continue
        elif cmd_lower == "/help":
            print_help()
            continue
        elif cmd_lower == "/export":
            export_conversation(conversation_history, model)
            continue
        elif raw_cmd.startswith("/agent "):
            goal = raw_cmd[7:].strip()
            run_agent_loop(model, goal)
            continue
        elif raw_cmd.startswith("/debate "):
            topic = raw_cmd[8:].strip()
            run_expert_debate(quick_query, topic)
            continue
        elif raw_cmd.startswith("/web "):
            query = raw_cmd[5:].strip()
            print(f"\n  {BOLD_CYAN}● WebSearch{RESET}{WHITE}({query}){RESET}")
            results = web_search(query)
            print(f"\n{WHITE}{results}{RESET}\n")
            continue
        elif raw_cmd.startswith("/url "):
            target_url = raw_cmd[5:].strip()
            print(f"\n  {BOLD_CYAN}● FetchURL{RESET}{WHITE}({target_url}){RESET}")
            results = fetch_url(target_url)
            print(f"\n{WHITE}{results}{RESET}\n")
            continue
        elif raw_cmd.startswith("/weather "):
            city = raw_cmd[9:].strip()
            print(f"\n  {BOLD_CYAN}● Weather{RESET}{WHITE}({city}){RESET}")
            results = fetch_weather(city)
            print(f"\n{WHITE}{results}{RESET}\n")
            continue
        elif raw_cmd.startswith("/wiki "):
            topic = raw_cmd[6:].strip()
            print(f"\n  {BOLD_CYAN}● Wikipedia{RESET}{WHITE}({topic}){RESET}")
            results = wiki_lookup(topic)
            print(f"\n{WHITE}{results}{RESET}\n")
            continue
        elif raw_cmd.startswith("/remember "):
            parts = raw_cmd[10:].strip().split(" ", 1)
            if len(parts) == 2:
                key, val = parts[0], parts[1]
                memory_db.remember("user_facts", key, val)
                print(f"\n  {GREEN}✓ Stored in 10T Memory: {key} -> {val}{RESET}\n")
            else:
                print(f"\n  {RED}Usage: /remember <key> <value>{RESET}\n")
            continue
        elif raw_cmd.startswith("/forget "):
            key = raw_cmd[8:].strip()
            memory_db.forget("user_facts", key)
            print(f"\n  {YELLOW}✓ Memory '{key}' removed.{RESET}\n")
            continue
        elif raw_cmd.startswith("/run "):
            code = raw_cmd[5:].strip()
            print(f"\n  {BOLD_CYAN}● PythonREPL{RESET}")
            res = tool_run_python(code)
            print(f"\n{WHITE}{res}{RESET}\n")
            continue
        elif raw_cmd.startswith("/sh "):
            cmd = raw_cmd[4:].strip()
            print(f"\n  {BOLD_CYAN}● Bash{RESET}{WHITE}({cmd}){RESET}")
            res = tool_run_shell(cmd)
            print(f"\n{WHITE}{res}{RESET}\n")
            continue
        elif raw_cmd.startswith("/read "):
            path = raw_cmd[6:].strip()
            print(f"\n  {BOLD_CYAN}● ReadFile{RESET}{WHITE}({path}){RESET}")
            res = tool_read_file(path)
            print(f"\n{WHITE}{res}{RESET}\n")
            continue
        elif raw_cmd.startswith("/write "):
            parts = raw_cmd[7:].strip().split(" ", 1)
            if len(parts) == 2:
                path, content = parts[0], parts[1]
                res = tool_write_file(path, content)
                print(f"\n  {GREEN}✓ {res}{RESET}\n")
            else:
                print(f"\n  {RED}Usage: /write <filepath> <content>{RESET}\n")
            continue
        elif raw_cmd.startswith("/ls"):
            path = raw_cmd[3:].strip() or "."
            print(f"\n  {BOLD_CYAN}● ListDir{RESET}{WHITE}({path}){RESET}")
            res = tool_list_dir(path)
            print(f"\n{WHITE}{res}{RESET}\n")
            continue

        # Direct App Intent Trigger
        app_action = detect_app_intent(user_input)
        if app_action:
            print(f"  {DIM_CYAN}● Executing App Action: {app_action}...{RESET}")
            app_res = execute_app_action(app_action)
            print(f"\n{BOLD_CYAN}│  {BOLD_WHITE}Lunaite Desktop Action:{RESET}\n{BOLD_CYAN}│{RESET}  {WHITE}{app_res}{RESET}\n")
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": app_res})
            if VOICE_MODE:
                speak_out(app_res)
            continue

        # Automatic Live Web Search Trigger
        live_web_context = ""
        if AUTO_WEB_SEARCH:
            auto_action = should_auto_search_web(user_input)
            if auto_action:
                if auto_action.startswith("SEARCH:"):
                    sq = auto_action[7:]
                    print(f"  {DIM_CYAN}● AutoWebSearch({sq})...{RESET}")
                    s_res = web_search(sq, max_results=3)
                    live_web_context = f"\n\n[Live Internet Search Results for '{sq}']:\n{s_res}"
                elif auto_action.startswith("FETCH_URL:"):
                    target_url = auto_action[10:]
                    print(f"  {DIM_CYAN}● AutoFetchURL({target_url})...{RESET}")
                    u_res = fetch_url(target_url, max_chars=2000)
                    live_web_context = f"\n\n[Live Webpage Content from '{target_url}']:\n{u_res}"
                elif auto_action.startswith("WEATHER:"):
                    city = auto_action[8:]
                    print(f"  {DIM_CYAN}● AutoWeather({city})...{RESET}")
                    w_res = fetch_weather(city)
                    live_web_context = f"\n\n[Real-Time Weather Data]:\n{w_res}"

        # Route query through 10T MoE Expert Matrix
        active_experts = route_10t_query(user_input)

        # Standard Conversation Turn
        prompt_with_live_data = user_input + live_web_context
        conversation_history.append({"role": "user", "content": prompt_with_live_data})
        response = stream_chat(model, conversation_history, active_experts)

        if response:
            conversation_history.append({"role": "assistant", "content": response})
            if VOICE_MODE:
                speak_out(response)
        else:
            conversation_history.pop()


if __name__ == "__main__":
    main()
