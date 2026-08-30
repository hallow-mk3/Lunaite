"""
lunaite.tools.registry
======================
ToolRegistry — a flat, named collection of Tool objects.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from lunaite.tools.tool import Tool


class ToolRegistry:
    """Holds all tools available to the agent.

    Tools are stored by name; duplicate names raise ``ValueError``.
    The registry is intentionally **not** thread-safe — it is built once
    at startup and treated as immutable during a run.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    # ------------------------------------------------------------------ #
    # Mutation (setup time only)                                           #
    # ------------------------------------------------------------------ #

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry.

        Raises
        ------
        ValueError
            If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named {tool.name!r} is already registered. "
                "Tool names must be unique."
            )
        self._tools[tool.name] = tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        """Convenience method to register multiple tools at once."""
        for tool in tools:
            self.register(tool)

    # ------------------------------------------------------------------ #
    # Lookup                                                               #
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> Optional[Tool]:
        """Return the tool with the given name, or ``None`` if not found."""
        return self._tools.get(name)

    def get_strict(self, name: str) -> Tool:
        """Return the tool with the given name.

        Raises
        ------
        KeyError
            If no tool with that name is registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(
                f"No tool named {name!r} in registry. "
                f"Available tools: {list(self._tools)}"
            )

    # ------------------------------------------------------------------ #
    # Read-only views                                                      #
    # ------------------------------------------------------------------ #

    def all_tools(self) -> List[Tool]:
        """Return all registered tools as a list (insertion order)."""
        return list(self._tools.values())

    def subset(self, names: Iterable[str]) -> List[Tool]:
        """Return the tools matching the given names (preserving order).

        Unknown names are silently skipped so callers can pass arbitrary
        scored lists without crashing on stale names.
        """
        return [self._tools[n] for n in names if n in self._tools]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:  # pragma: no cover
        return f"ToolRegistry({len(self)} tools: {list(self._tools)})"
