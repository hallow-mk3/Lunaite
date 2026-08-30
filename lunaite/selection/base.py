"""
lunaite.selection.base
======================
Abstract base class for all tool selectors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from lunaite.tools.registry import ToolRegistry
from lunaite.tools.tool import Tool


class Selector(ABC):
    """Interface every selector must implement.

    A selector receives a natural-language *query* and a *registry* of all
    available tools, and returns the subset of tools that should be shown
    to the LLM for that query.
    """

    @abstractmethod
    def select(self, query: str, registry: ToolRegistry, **kwargs) -> List[Tool]:
        """Select a subset of tools relevant to *query*.

        Parameters
        ----------
        query:
            The user's natural-language request.
        registry:
            The full collection of available tools.
        **kwargs:
            Selector-specific options (e.g. ``k`` for retrieval selectors).

        Returns
        -------
        List[Tool]
            The tools to expose to the LLM.  May be all tools (naive) or a
            scored subset (retrieval).  Order is meaningful: index 0 is most
            relevant.
        """
        ...

    def name(self) -> str:  # pragma: no cover
        """Human-readable name used in logs and reports."""
        return self.__class__.__name__
