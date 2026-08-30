"""
lunaite.selection.naive
=======================
NaiveSelector — returns every tool in the registry, unfiltered.

This is the baseline that retrieval selection must beat.  Most real
function-calling systems today use this strategy (dump all schemas into
context), which makes it the practically relevant comparison point.
"""
from __future__ import annotations

from typing import List

from lunaite.selection.base import Selector
from lunaite.tools.registry import ToolRegistry
from lunaite.tools.tool import Tool


class NaiveSelector(Selector):
    """Return all tools in the registry, unfiltered and in insertion order."""

    def select(self, query: str, registry: ToolRegistry, **kwargs) -> List[Tool]:
        # query is intentionally unused — this is the "no selection" baseline.
        return registry.all_tools()

    def name(self) -> str:
        return "naive"
