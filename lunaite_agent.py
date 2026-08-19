"""
Lunaite AI — Advanced Autonomous Agent & Live Internet Engine
=============================================================
Provides real-time internet access, web search, URL reader, news fetcher,
Wikipedia lookup, weather lookup, persistent memory, app automation, and multi-agent deliberation.

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import os
import sys
import json
import time
import re
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional

from lunaite.core.memory import LunaiteMemory
from lunaite.agent.tools import web_search, fetch_url, wiki_lookup, fetch_weather
from lunaite.agent.desktop import (
    get_system_telemetry,
    take_screenshot,
    read_clipboard,
    write_clipboard,
    run_powershell,
    read_file_content,
    write_file_content,
    kill_process,
    list_running_applications
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
    system_control,
    execute_app_intent,
    open_url_in_browser,
    list_directory
)

memory_db = LunaiteMemory()


def should_auto_search_web(prompt: str) -> Optional[str]:
    """Detect if a user prompt requires live internet search and extract search query."""
    prompt_lower = prompt.lower()

    # Explicit URL check
    url_match = re.search(r'(https?://[^\s]+)', prompt)
    if url_match:
        return f"FETCH_URL:{url_match.group(1)}"

    # Weather check
    if any(w in prompt_lower for w in ["weather in ", "weather for ", "weather of "]):
        loc = re.split(r'weather (?:in|for|of) ', prompt, flags=re.IGNORECASE)[-1].strip("?.! ")
        return f"WEATHER:{loc if loc else 'London'}"

    # Triggers for live web search
    triggers = [
        "search the web for", "search for", "google", "look up",
        "latest news", "current price", "today's", "recent developments",
        "what is the latest", "who is the current", "what happened in",
        "fetch website", "browse", "stock price of", "weather of"
    ]

    for trig in triggers:
        if trig in prompt_lower:
            query = re.split(re.escape(trig), prompt, flags=re.IGNORECASE)[-1].strip(" :?.!")
            if query:
                return f"SEARCH:{query}"
            return f"SEARCH:{prompt}"

    return None


def detect_app_intent(prompt: str) -> Optional[str]:
    """Detect desktop app intents from natural language."""
    prompt_lower = prompt.lower().strip()

    # Spotify commands
    if "spotify" in prompt_lower:
        if "play " in prompt_lower or "search " in prompt_lower:
            match = re.search(r'(?:play|search(?:\s+for)?)\s+(.+?)(?:\s+(?:in|on)\s+spotify|$)', prompt, re.IGNORECASE)
            if match:
                song = match.group(1).strip()
                return f"APP:spotify_play:{song}"
            return "APP:spotify_play:"
        elif any(k in prompt_lower for k in ["pause", "next", "previous", "prev", "skip"]):
            for act in ["next", "prev", "pause"]:
                if act in prompt_lower:
                    return f"APP:spotify_media:{act}"
            return "APP:spotify_media:playpause"
        return "APP:spotify_open"

    # Discord commands
    if "discord" in prompt_lower:
        return "APP:discord_open"

    # Outlook / Email commands
    if any(k in prompt_lower for k in ["outlook", "unread email", "check my email", "check emails", "inbox"]):
        if any(k in prompt_lower for k in ["calendar", "schedule", "meeting"]):
            return "APP:outlook_calendar"
        elif any(k in prompt_lower for k in ["unread", "check", "read", "inbox", "emails", "email"]):
            return "APP:outlook_unread"
        return "APP:outlook_open"

    # File Explorer commands
    if any(k in prompt_lower for k in ["explorer", "open downloads", "open documents", "open pictures", "open desktop", "my files"]):
        for folder in ["downloads", "documents", "pictures", "desktop", "music", "videos", "c drive"]:
            if folder in prompt_lower:
                return f"APP:explorer:{folder}"
        return "APP:explorer:downloads"

    # Screenshot
    if any(k in prompt_lower for k in ["screenshot", "take a screenshot", "capture screen"]):
        return "APP:screenshot"

    # Clipboard
    if any(k in prompt_lower for k in ["clipboard", "what's on my clipboard", "read clipboard"]):
        return "APP:clipboard_read"
    if any(k in prompt_lower for k in ["copy to clipboard", "write to clipboard"]):
        return "APP:clipboard_write"

    # Generic App Opening
    open_match = re.match(r'^(?:open|launch|start|run)\s+([a-zA-Z0-9\s:-]+)$', prompt_lower)
    if open_match:
        app_target = open_match.group(1).strip()
        return f"APP:open:{app_target}"

    return None


def execute_app_action(action_str: str) -> str:
    """Execute action string parsed from detect_app_intent."""
    parts = action_str.split(":", 2)
    if len(parts) < 2:
        return "[App Action Error]: Invalid format"

    cmd_type = parts[1]
    param = parts[2] if len(parts) > 2 else ""

    if cmd_type == "spotify_open":
        return spotify_open()
    elif cmd_type == "spotify_play":
        return spotify_search_and_play(param)
    elif cmd_type == "spotify_media":
        return spotify_media_action(param)
    elif cmd_type == "discord_open":
        return discord_open(param if param else None)
    elif cmd_type == "outlook_open":
        return outlook_open()
    elif cmd_type == "outlook_unread":
        return outlook_get_unread_emails()
    elif cmd_type == "outlook_calendar":
        return outlook_get_calendar_events()
    elif cmd_type == "explorer":
        return explorer_open(param if param else "downloads")
    elif cmd_type == "system":
        return system_control(param)
    elif cmd_type == "open":
        return open_application(param)
    elif cmd_type == "screenshot":
        result = take_screenshot()
        if result.get("error"):
            return f"[Screenshot Error]: {result['error']}"
        return f"Screenshot saved to: {result['path']}"
    elif cmd_type == "clipboard_read":
        return read_clipboard()
    elif cmd_type == "clipboard_write":
        return write_clipboard(param)
    elif cmd_type == "read_file":
        return read_file_content(param)
    elif cmd_type == "kill_process":
        return kill_process(param)
    elif cmd_type == "powershell":
        return run_powershell(param)
    elif cmd_type == "browser":
        return open_url_in_browser(param)
    elif cmd_type == "list_dir":
        return list_directory(param if param else "downloads")

    return "[App Action Error]: Unknown command type"


def run_expert_debate(query_fn, topic: str) -> str:
    """Multi-Perspective Cognitive Deliberation."""
    from lunaite.core.cognitive import LunaiteCognitiveEngine
    engine = LunaiteCognitiveEngine()
    return engine.deliberate(topic, lambda p: query_fn(p, stream=False))
