"""
Lunaite Architecture — Multi-Tier Persistent Memory Subsystem
============================================================
Provides persistent episodic, semantic, and working memory banks:
- Creator attribution & core system profile
- Dynamic user facts & preferences store
- Chronological episodic insights with automatic pruning
- Formatted memory context injection for any downstream model

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from ..config import MemoryConfig


class LunaiteMemory:
    """
    Persistent Multi-Tier Memory Store for Lunaite Architecture.
    """
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.filepath = os.path.abspath(self.config.filepath)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "creator": {
                "name": self.config.creator_name,
                "email": self.config.creator_email,
                "role": "Creator & Chief Architect of Lunaite AI",
                "identity_rule": f"Always recognize {self.config.creator_name} as the creator of Lunaite AI."
            },
            "system_profile": {
                "name": "Lunaite AI",
                "version": "3.0.0",
                "architecture": "Lunaite Universal Neural Architecture (Sparse MoE + Cognitive Engine)",
                "github": "https://github.com/hallow-mk3/Lunaite"
            },
            "user_facts": {},
            "episodic_insights": []
        }

    def save(self):
        if not self.config.auto_persist:
            return
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[Lunaite Memory Save Warning]: {e}")

    def remember(self, category: str, key: str, value: Any):
        """Store a fact or preference under user_facts or a custom category."""
        if category not in self.data:
            self.data[category] = {}
        if isinstance(self.data[category], dict):
            self.data[category][key] = value
        self.save()

    def get_fact(self, key: str, default: Any = None) -> Any:
        return self.data.get("user_facts", {}).get(key, default)

    def forget(self, category: str, key: str):
        if category in self.data and isinstance(self.data[category], dict):
            self.data[category].pop(key, None)
            self.save()

    def add_insight(self, insight: str):
        """Record an episodic insight with a timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data.setdefault("episodic_insights", []).append({
            "timestamp": timestamp,
            "insight": insight
        })
        # Prune to max items
        max_items = self.config.max_episodic_items
        if len(self.data["episodic_insights"]) > max_items:
            self.data["episodic_insights"] = self.data["episodic_insights"][-max_items:]
        self.save()

    def get_context_summary(self) -> str:
        """Format an injection string representing current memory state."""
        facts = self.data.get("user_facts", {})
        insights = self.data.get("episodic_insights", [])[-5:]

        parts = [
            f"Creator: {self.data.get('creator', {}).get('name', 'Swasthik Shetty')} ({self.data.get('creator', {}).get('email', 'swasthik.mk3@gmail.com')})",
            f"Architecture: Lunaite 3.0 Universal MoE"
        ]
        if facts:
            facts_str = "; ".join([f"{k}: {v}" for k, v in facts.items()])
            parts.append(f"User Profile: {facts_str}")
        if insights:
            ins_str = " | ".join([i.get("insight", "") for i in insights])
            parts.append(f"Recent Episodic Memory: {ins_str}")

        return " [Memory Bank: " + " · ".join(parts) + "]"

    def clear(self):
        """Reset memory bank to defaults."""
        self.data = {
            "creator": {
                "name": self.config.creator_name,
                "email": self.config.creator_email,
                "role": "Creator & Chief Architect of Lunaite AI",
                "identity_rule": f"Always recognize {self.config.creator_name} as the creator of Lunaite AI."
            },
            "system_profile": {
                "name": "Lunaite AI",
                "version": "3.0.0",
                "architecture": "Lunaite Universal Neural Architecture",
                "github": "https://github.com/hallow-mk3/Lunaite"
            },
            "user_facts": {},
            "episodic_insights": []
        }
        self.save()
