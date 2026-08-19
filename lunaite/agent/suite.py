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


class LunaiteAgent:
    """
    Autonomous tool detection and execution engine for Lunaite Architecture.
    """
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def detect_intent(self, prompt: str) -> Optional[Tuple[str, str]]:
        """
        Analyze user query to detect if a live tool or system action is needed.
        Returns:
            (tool_name, argument) or None
        """
        prompt_lower = prompt.lower().strip()

        # 1. URL fetch
        url_match = re.search(r'(https?://[^\s]+)', prompt)
        if url_match:
            return ("fetch_url", url_match.group(1))

        # 2. Weather
        if any(w in prompt_lower for w in ["weather in ", "weather for ", "weather of "]):
            loc = re.split(r'weather (?:in|for|of) ', prompt, flags=re.IGNORECASE)[-1].strip("?.! ")
            return ("weather", loc if loc else "London")

        # 3. Screenshot
        if any(k in prompt_lower for k in ["take a screenshot", "capture screen", "screenshot"]):
            return ("screenshot", "")

        # 4. Clipboard
        if any(k in prompt_lower for k in ["read clipboard", "what's on my clipboard", "clipboard content"]):
            return ("clipboard_read", "")

        # 5. System Telemetry
        if any(k in prompt_lower for k in ["system vitals", "system telemetry", "hardware stats", "ram usage", "cpu usage"]):
            return ("telemetry", "")

        # 6. Live Web Search Triggers
        if self.config.auto_web_search:
            search_triggers = [
                "search the web for", "search for", "google", "look up",
                "latest news", "current price", "today's", "recent developments",
                "what is the latest", "who is the current", "what happened in",
                "browse"
            ]
            for trig in search_triggers:
                if trig in prompt_lower:
                    query = re.split(re.escape(trig), prompt, flags=re.IGNORECASE)[-1].strip(" :?.!")
                    return ("web_search", query if query else prompt)

        return None

    def execute_tool(self, tool_name: str, arg: str) -> str:
        """Execute a detected tool action."""
        if tool_name == "web_search":
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
