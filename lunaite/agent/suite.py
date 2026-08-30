"""
Lunaite Architecture — Universal Agent & Tool Orchestration Suite
=================================================================
Orchestrates autonomous intent detection, web search, system telemetry,
and desktop execution with downstream neural models.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Callable
from ..config import AgentConfig
from .tools import web_search, fetch_url, wiki_lookup, fetch_weather
from .desktop import (
    get_system_telemetry,
    take_screenshot,
    read_clipboard,
    write_clipboard,
    run_powershell,
    read_file_content,
    write_file_content,
    list_running_applications,
    kill_process
)


TOOL_SYSTEM_PROMPT = """You are Lunaite's Tool Orchestration Router.
Analyze the user's input and decide if an external tool or real-time action is required.
Available Tools:
- web_search(query): Real-time web search for recent events, news, current facts, launches, future schedules, stock prices, external info.
- get_current_time(location): Look up current time, timezone, or date.
- weather(location): Real-time weather forecast for any city or region.
- wiki(topic): Detailed encyclopedia/biography/historical knowledge.
- read_file(path): Inspect content of a local file.
- write_file(path, content): Create or overwrite a local file.
- powershell(command): Execute a terminal / shell command and return stdout/stderr.
- screenshot(): Capture a display screenshot.
- telemetry(): Get current CPU, RAM, GPU, and Disk stats.
- none: If the question can be answered purely from conversation, mathematics, general knowledge, or coding logic.

Decision Rule:
Output strictly JSON in this exact schema without any extra commentary:
{"tool": "web_search" | "get_current_time" | "weather" | "wiki" | "read_file" | "write_file" | "powershell" | "screenshot" | "telemetry" | "none", "arg": "<argument string or empty>"}
"""


def get_current_time_str(location: str = "") -> str:
    """Return formatted current time and date."""
    loc_clean = location.strip().lower()
    now_utc = time.gmtime()
    if "india" in loc_clean or "ist" in loc_clean:
        # UTC+5:30
        ist_timestamp = time.time() + 5.5 * 3600
        ist_struct = time.gmtime(ist_timestamp)
        return f"Current Time in India (IST, UTC+5:30): {time.strftime('%Y-%m-%d %I:%M:%S %p (%A)', ist_struct)}"
    
    local_t = time.localtime()
    return f"Current System Time: {time.strftime('%Y-%m-%d %I:%M:%S %p (%A %Z)', local_t)} (UTC: {time.strftime('%Y-%m-%d %H:%M:%S UTC', now_utc)})"


class LunaiteAgent:
    """
    Autonomous tool detection and execution engine for Lunaite Architecture.
    Combines rapid heuristic routing with LLM-driven autonomous tool decisions.
    """
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def decide_tool(self, prompt: str, generate_fn: Optional[Callable[[str], str]] = None) -> Optional[Tuple[str, str]]:
        """
        Dynamically determine if a tool is needed using fast heuristic pattern matching
        with fallback to LLM tool deliberation.
        """
        prompt_lower = prompt.lower().strip()

        # 1. Direct Time / Date Queries
        if any(k in prompt_lower for k in ["time in ", "current time", "what time is it", "today's date", "time right now"]):
            loc = re.split(r'time (?:in|for|at|of) ', prompt, flags=re.IGNORECASE)[-1].strip("?.! ")
            return ("time", loc if loc else "")

        # 2. Direct Weather
        if any(w in prompt_lower for w in ["weather in ", "weather for ", "weather of ", "temperature in"]):
            loc = re.split(r'(?:weather|temperature) (?:in|for|of) ', prompt, flags=re.IGNORECASE)[-1].strip("?.! ")
            return ("weather", loc if loc else "London")

        # 3. Direct URL Fetch
        url_match = re.search(r'(https?://[^\s]+)', prompt)
        if url_match:
            return ("fetch_url", url_match.group(1))

        # 4. File Reading
        file_read_match = re.search(r'(?:read|open|show|view|inspect)\s+(?:the\s+)?(?:file|code)?\s*([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9_]+)', prompt, re.IGNORECASE)
        if file_read_match:
            return ("read_file", file_read_match.group(1))

        # 5. File Writing
        file_write_match = re.search(r'(?:create|write|save)\s+(?:file|to)?\s*([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9_]+)\s*(?:with|as)?\s*[:\n](.+)', prompt, re.IGNORECASE | re.DOTALL)
        if file_write_match:
            return ("write_file", f"{file_write_match.group(1)} ::: {file_write_match.group(2)}")

        # 6. Shell / PowerShell
        cmd_match = re.search(r'(?:run command|execute command|run shell|run powershell|execute)\s*[:\s]+`?([^`\n]+)`?', prompt, re.IGNORECASE)
        if cmd_match:
            return ("powershell", cmd_match.group(1).strip())

        # 7. System Vitals / Screenshot / Clipboard
        if any(k in prompt_lower for k in ["system vitals", "system telemetry", "hardware stats", "ram usage", "cpu usage"]):
            return ("telemetry", "")
        if any(k in prompt_lower for k in ["take a screenshot", "capture screen"]):
            return ("screenshot", "")
        if any(k in prompt_lower for k in ["read clipboard", "clipboard content"]):
            return ("clipboard_read", "")

        # 8. Web Search Heuristics (news, current events, recent developments, schedules, launches)
        search_triggers = [
            "latest", "recent", "news", "current", "when is", "what is the price",
            "upcoming", "launching", "schedule", "who won", "score", "today",
            "search the web", "search for", "google", "look up"
        ]
        if self.config.auto_web_search and any(trig in prompt_lower for trig in search_triggers):
            # Clean search query
            clean_q = re.sub(r'^(?:search\s+for|search\s+the\s+web\s+for|google|look\s+up|what\s+is\s+the\s+|when\s+is\s+the\s+|who\s+is\s+the\s+)', '', prompt, flags=re.IGNORECASE).strip("?.! ")
            return ("web_search", clean_q if clean_q else prompt)

        # 9. LLM-Driven Autonomous Tool Decision (when uncertain and generate_fn provided)
        if generate_fn and self.config.auto_web_search:
            try:
                decide_prompt = f"{TOOL_SYSTEM_PROMPT}\nUser Query: {prompt}\nJSON Decision:"
                raw_decision = generate_fn(decide_prompt).strip()
                # Parse JSON block
                json_match = re.search(r'\{.*\}', raw_decision, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    tool = data.get("tool", "none").lower()
                    arg = data.get("arg", "").strip()
                    if tool and tool != "none":
                        if tool == "get_current_time":
                            return ("time", arg)
                        return (tool, arg if arg else prompt)
            except Exception:
                pass

        return None

    def detect_intent(self, prompt: str) -> Optional[Tuple[str, str]]:
        """Alias for backward compatibility."""
        return self.decide_tool(prompt)

    def execute_tool(self, tool_name: str, arg: str) -> str:
        """Execute a detected tool action."""
        if tool_name == "time":
            return get_current_time_str(arg)
        elif tool_name == "web_search":
            return web_search(arg, max_results=self.config.max_search_results)
        elif tool_name == "fetch_url":
            return fetch_url(arg, max_chars=self.config.max_url_chars)
        elif tool_name == "weather":
            return fetch_weather(arg)
        elif tool_name == "wiki":
            return wiki_lookup(arg)
        elif tool_name == "telemetry":
            stats = get_system_telemetry()
            return f"**System Telemetry:**\n- CPU: {stats['cpu_percent']}%\n- RAM: {stats['ram_used_gb']}GB / {stats['ram_total_gb']}GB ({stats['ram_percent']}%)\n- GPU: {stats['gpu_name']} ({stats['gpu_vram_used_gb']}GB VRAM used)\n- Free Disk: {stats['disk_free_gb']}GB"
        elif tool_name == "screenshot":
            res = take_screenshot()
            return f"Screenshot taken: {res.get('path', res.get('error'))}"
        elif tool_name == "clipboard_read":
            return f"Clipboard: {read_clipboard()}"
        elif tool_name == "powershell":
            return run_powershell(arg)
        elif tool_name == "read_file":
            return read_file_content(arg)
        elif tool_name == "write_file":
            parts = arg.split(":::", 1)
            if len(parts) == 2:
                return write_file_content(parts[0].strip(), parts[1].strip())
            return "Invalid write_file syntax (expected 'path ::: content')"
        elif tool_name == "kill":
            return kill_process(arg)

        return f"Unknown tool: {tool_name}"

    def process_prompt(
        self,
        prompt: str,
        generate_fn: Callable[[str], str],
        on_tool_call: Optional[Callable[[str, str, str], None]] = None
    ) -> str:
        """
        End-to-end autonomous agent step:
        Detects if tools are required, executes them, injects evidence into prompt,
        and generates final model response.
        """
        intent = self.detect_intent(prompt)
        if intent is None:
            return generate_fn(prompt)

        tool_name, tool_arg = intent
        tool_result = self.execute_tool(tool_name, tool_arg)

        if on_tool_call:
            on_tool_call(tool_name, tool_arg, tool_result)

        augmented_prompt = (
            f"[Autonomous Tool Execution Result — {tool_name.upper()}]:\n"
            f"{tool_result}\n\n"
            f"User Instruction: {prompt}\n\n"
            f"Synthesize the tool findings and respond directly and insightfully to the user:"
        )
        return generate_fn(augmented_prompt)
