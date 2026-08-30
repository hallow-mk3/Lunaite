"""
lunaite.harness.harness
=======================
Harness — main orchestration class.

Given a user query, a ToolRegistry, and a Selector, the Harness:
  1. Selects a subset of tools via the selector.
  2. Calls the LLM (OpenAI-compatible / Ollama endpoint) with the query
     and selected tool schemas.
  3. Parses the tool call from the response.
  4. Executes the tool callable.
  5. Returns a structured HarnessResult with every step logged.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import requests

from lunaite.selection.base import Selector
from lunaite.tools.registry import ToolRegistry
from lunaite.tools.tool import Tool


# ------------------------------------------------------------------ #
# Result dataclass                                                     #
# ------------------------------------------------------------------ #


@dataclass
class HarnessResult:
    """Full record of one Harness.run() call.

    All fields are JSON-serialisable so callers can write directly to
    JSONL without custom encoders.
    """

    # Input
    query: str
    selector_name: str
    model: str

    # Selection step
    tools_shown: List[str] = field(default_factory=list)
    n_tools_shown: int = 0

    # LLM response
    raw_response: Optional[str] = None
    tool_called: Optional[str] = None
    arguments_generated: Optional[Dict[str, Any]] = None
    no_tool_called: bool = False

    # Execution
    execution_result: Optional[Any] = None
    execution_error: Optional[str] = None
    execution_success: bool = False

    # Cost / latency
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0

    # Catch-all error
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ------------------------------------------------------------------ #
# Harness                                                              #
# ------------------------------------------------------------------ #


class Harness:
    """Orchestrates query → select → LLM → parse → execute → log.

    Parameters
    ----------
    registry:
        The full tool registry for this session.
    selector:
        A :class:`~lunaite.selection.base.Selector` instance.
    model:
        Model name (e.g. ``"qwen2.5:7b"`` or ``"gpt-4o-mini"``).
    base_url:
        OpenAI-compatible base URL.
    api_key:
        Optional API key (defaults to ``OPENAI_API_KEY`` env variable).
    temperature:
        Sampling temperature for the LLM.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        selector: Selector,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> None:
        self.registry = registry
        self.selector = selector
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def check_connection(self) -> bool:
        """Check if the backend LLM endpoint is reachable."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            resp = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def run(self, query: str, selector_kwargs: Optional[Dict] = None) -> HarnessResult:
        """Execute one full query → tool-call cycle.

        Parameters
        ----------
        query:
            The user's natural-language request.
        selector_kwargs:
            Extra kwargs forwarded to ``selector.select()`` (e.g. ``{"k": 5}``).

        Returns
        -------
        HarnessResult
            A fully-populated result record.
        """
        selector_kwargs = selector_kwargs or {}
        result = HarnessResult(
            query=query,
            selector_name=self.selector.name(),
            model=self.model,
        )

        t_start = time.perf_counter()
        try:
            # ── Step 1: select tools ──────────────────────────────────── #
            selected_tools: List[Tool] = self.selector.select(
                query, self.registry, **selector_kwargs
            )
            result.tools_shown = [t.name for t in selected_tools]
            result.n_tools_shown = len(selected_tools)

            # ── Step 2: call LLM ─────────────────────────────────────── #
            api_response = self._call_llm(query, selected_tools)
            result.raw_response = json.dumps(api_response)

            # ── Step 3: parse usage ──────────────────────────────────── #
            usage = api_response.get("usage", {})
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)
            result.total_tokens = usage.get("total_tokens", 0)

            # ── Step 4: parse tool call ──────────────────────────────── #
            choices = api_response.get("choices", [])
            if not choices:
                result.no_tool_called = True
            else:
                choice = choices[0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls") or []

                if not tool_calls:
                    result.no_tool_called = True
                else:
                    tc = tool_calls[0]
                    fn = tc.get("function", {})
                    result.tool_called = fn.get("name")
                    raw_args = fn.get("arguments", "{}")
                    result.arguments_generated = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )

                    # ── Step 5: execute ───────────────────────────────────── #
                    tool = self.registry.get(result.tool_called)
                    if tool is None:
                        result.execution_error = (
                            f"Model called unknown tool: {result.tool_called!r}"
                        )
                    else:
                        try:
                            result.execution_result = tool.call(
                                **result.arguments_generated
                            )
                            result.execution_success = True
                        except Exception as exc:
                            result.execution_error = f"{type(exc).__name__}: {exc}"

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.latency_s = time.perf_counter() - t_start

        return result

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _call_llm(self, query: str, tools: List[Tool]) -> Dict:
        """Call the OpenAI-compatible /v1/chat/completions endpoint."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": query}],
            "temperature": self.temperature,
            "stream": False,
        }

        if tools:
            payload["tools"] = [t.to_openai_schema() for t in tools]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()
