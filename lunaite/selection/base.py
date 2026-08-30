"""
lunaite.selection.base
======================
Abstract base class for all tool selectors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from lunaite.tools.registry import ToolRegistry
from lunaite.tools.tool import Tool


class Selector(ABC):
    """Interface every selector must implement.

    A selector receives a natural-language *query* and a *registry* of all
    available tools, and returns the subset of tools that should be shown
    to the LLM for that query.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry

    @abstractmethod
    def select(self, query: str, registry: Optional[ToolRegistry] = None, **kwargs) -> List[Tool]:
        """Select a subset of tools relevant to *query*.

        Parameters
        ----------
        query:
            The user's natural-language request.
        registry:
            The full collection of available tools (optional if bound in __init__).
        **kwargs:
            Selector-specific options (e.g. ``k`` for retrieval selectors).

        Returns
        -------
        List[Tool]
            The tools to expose to the LLM. Order is meaningful: index 0 is most
            relevant.
        """
        ...

    def name(self) -> str:  # pragma: no cover
        """Human-readable name used in logs and reports."""
        return self.__class__.__name__
