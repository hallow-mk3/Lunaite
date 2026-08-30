"""
lunaite.tools.registry
======================
ToolRegistry — a flat, named collection of Tool objects.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

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

    def register(
        self,
        tool: Optional[Union[Tool, str]] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Add a tool to the registry, or use as a decorator.

        Usage:
            # 1. Direct Tool instance
            registry.register(my_tool)

            # 2. Decorator syntax
            @registry.register(name="get_weather", description="Fetch weather", parameters={...})
            def get_weather(city: str):
                ...
        """
        # Case 1: Direct Tool instance passed as positional argument
        if isinstance(tool, Tool):
            if tool.name in self._tools:
                raise ValueError(
                    f"A tool named {tool.name!r} is already registered. "
                    "Tool names must be unique."
                )
            self._tools[tool.name] = tool
            return tool

        # Case 2: Decorator syntax
        tool_name = name or (tool if isinstance(tool, str) else None)

        def decorator(fn: Callable[..., Any]) -> Tool:
            final_name = tool_name or fn.__name__
            final_desc = description or (fn.__doc__ or "").strip() or f"Tool {final_name}"
            final_params = parameters or {}
            
            # If parameters is a simplified dict without 'type': 'object', wrap it
            if "type" not in final_params or final_params.get("type") != "object":
                schema_params = {
                    "type": "object",
                    "properties": final_params.get("properties", final_params),
                }
                if "required" in final_params:
                    schema_params["required"] = final_params["required"]
            else:
                schema_params = final_params

            new_tool = Tool(
                name=final_name,
                description=final_desc,
                parameters=schema_params,
                callable=fn,
            )
            self.register(new_tool)
            return new_tool

        return decorator

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
