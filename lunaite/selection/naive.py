"""
lunaite.selection.naive
=======================
NaiveSelector — returns every tool in the registry, unfiltered.

This is the baseline that retrieval selection must beat. Most real
function-calling systems today use this strategy (dump all schemas into
context), which makes it the practically relevant comparison point.
"""
from __future__ import annotations

from typing import List, Optional

from lunaite.selection.base import Selector
from lunaite.tools.registry import ToolRegistry
from lunaite.tools.tool import Tool


class NaiveSelector(Selector):
    """Return all tools in the registry, unfiltered and in insertion order."""

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        super().__init__(registry)

    def select(self, query: str, registry: Optional[ToolRegistry] = None, **kwargs) -> List[Tool]:
        reg = registry or self.registry
        if reg is None:
            raise ValueError("ToolRegistry must be provided either at init or in select()")
        # query is intentionally unused — this is the "no selection" baseline.
        return reg.all_tools()

    def name(self) -> str:
        return "naive"
