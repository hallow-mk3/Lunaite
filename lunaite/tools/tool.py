"""
lunaite.tools.tool
==================
Core Tool definition.

Security stance
---------------
Only read-only / non-destructive callables belong in a ToolRegistry.
Shell execution, filesystem writes, and clipboard access are explicitly
excluded — this is a deliberate security boundary, not an oversight.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass
class Tool:
    """A single tool that the LLM can call.

    Attributes
    ----------
    name:
        Unique identifier used in the LLM's function-call JSON.
    description:
        Natural-language description used both for prompting and for
        embedding-based retrieval.  Write it as a one-sentence answer to
        "What does this tool do?".
    parameters:
        JSON Schema object (dict) describing the tool's arguments.
        Must be a valid ``object`` schema with a ``properties`` key.
    callable:
        The Python function that executes the tool.  Must accept keyword
        arguments matching the JSON schema and return a JSON-serialisable
        value.  Raise ``ValueError`` for bad inputs, ``RuntimeError`` for
        execution failures.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    callable: Callable[..., Any]

    # ------------------------------------------------------------------ #
    # Derived helpers                                                       #
    # ------------------------------------------------------------------ #

    def to_openai_schema(self) -> Dict[str, Any]:
        """Return the tool schema in OpenAI / Ollama function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, **kwargs: Any) -> Any:
        """Execute the tool with validated kwargs and return the result."""
        return self.callable(**kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Tool(name={self.name!r})"
