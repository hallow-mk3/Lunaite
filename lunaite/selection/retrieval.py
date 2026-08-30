"""
lunaite.selection.retrieval
===========================
RetrievalSelector — embed query and tool descriptions, return top-k by
cosine similarity.

Embedding model: ``all-MiniLM-L6-v2`` (sentence-transformers).
Similarity: cosine via numpy (no external vector DB needed at ≤50 tools).

The model is loaded once on first use (lazy) and cached for the lifetime
of the selector instance to avoid repeated disk I/O.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

from lunaite.selection.base import Selector
from lunaite.tools.registry import ToolRegistry
from lunaite.tools.tool import Tool

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_K = 5


class RetrievalSelector(Selector):
    """Select the top-*k* tools most similar to the query via embedding cosine.

    Parameters
    ----------
    model_name:
        A sentence-transformers model identifier.  Defaults to
        ``all-MiniLM-L6-v2`` (22 MB, fast CPU inference).
    k:
        Default number of tools to return when ``k`` is not passed to
        :meth:`select`.  Can be overridden per-call via ``select(..., k=N)``.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        k: int = _DEFAULT_K,
    ) -> None:
        self._model_name = model_name
        self._default_k = k
        self._model = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get_model(self):
        """Lazy-load the embedding model (thread-safe)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self._model_name)
        return self._model

    @staticmethod
    def _cosine_similarities(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Return cosine similarities between *query_vec* and each row of *matrix*."""
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        return matrix_norms @ query_norm  # shape: (n_tools,)

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def select(self, query: str, registry: ToolRegistry, **kwargs) -> List[Tool]:
        """Return the top-k tools most similar to *query*.

        Parameters
        ----------
        query:
            The user's natural-language request.
        registry:
            Full tool registry.
        k:
            Number of tools to return (overrides instance default).
        """
        k: int = kwargs.get("k", self._default_k)
        tools = registry.all_tools()

        if len(tools) == 0:
            return []

        # Clamp k to the number of available tools.
        k = min(k, len(tools))

        model = self._get_model()

        # Embed query and all descriptions in one batch for efficiency.
        descriptions = [t.description for t in tools]
        all_texts = [query] + descriptions
        embeddings = model.encode(all_texts, convert_to_numpy=True)

        query_vec: np.ndarray = embeddings[0]
        tool_matrix: np.ndarray = embeddings[1:]

        scores = self._cosine_similarities(query_vec, tool_matrix)

        # Sort descending by score, return top-k.
        top_indices = np.argsort(scores)[::-1][:k]
        return [tools[i] for i in top_indices]

    def name(self) -> str:
        return f"retrieval({self._model_name})"
