"""
lunaite.selection.retrieval
===========================
RetrievalSelector — embed query and tool descriptions, return top-k by
similarity.

Embedding model: ``all-MiniLM-L6-v2`` (sentence-transformers) with graceful
token-overlap fallback if sentence-transformers is not installed.
"""
from __future__ import annotations

import re
import threading
from typing import List, Optional, Union

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
    registry:
        Optional ToolRegistry instance to bind to.
    model_name:
        A sentence-transformers model identifier. Defaults to
        ``all-MiniLM-L6-v2``.
    k:
        Default number of tools to return.
    """

    def __init__(
        self,
        registry_or_model: Optional[Union[ToolRegistry, str]] = None,
        model_name: str = _DEFAULT_MODEL,
        k: int = _DEFAULT_K,
        **kwargs,
    ) -> None:
        if isinstance(registry_or_model, ToolRegistry):
            super().__init__(registry_or_model)
            self._model_name = kwargs.get("model_name", model_name)
            self._default_k = kwargs.get("k", k)
        elif isinstance(registry_or_model, str):
            super().__init__(kwargs.get("registry"))
            self._model_name = registry_or_model
            self._default_k = kwargs.get("k", k)
        else:
            super().__init__(kwargs.get("registry"))
            self._model_name = model_name
            self._default_k = kwargs.get("k", k)

        self._model = None
        self._has_sentence_transformers: Optional[bool] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get_model(self):
        """Lazy-load the embedding model if installed."""
        if self._has_sentence_transformers is False:
            return None

        if self._model is None:
            with self._lock:
                if self._model is None and self._has_sentence_transformers is not False:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer(self._model_name)
                        self._has_sentence_transformers = True
                    except Exception:
                        self._has_sentence_transformers = False
                        self._model = None
        return self._model

    @staticmethod
    def _cosine_similarities(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Return cosine similarities between *query_vec* and each row of *matrix*."""
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        return matrix_norms @ query_norm  # shape: (n_tools,)

    @staticmethod
    def _keyword_similarity_scores(query: str, tools: List[Tool]) -> np.ndarray:
        """Fallback lexical similarity when sentence-transformers is unavailable."""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        scores = []
        for t in tools:
            target_text = f"{t.name} {t.description} {' '.join(t.parameters.get('properties', {}).keys())}".lower()
            target_tokens = set(re.findall(r"\w+", target_text))
            intersection = query_tokens.intersection(target_tokens)
            score = len(intersection) / max(1, len(query_tokens))
            scores.append(score)
        return np.array(scores, dtype=float)

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def select(
        self,
        query: str,
        registry: Optional[ToolRegistry] = None,
        **kwargs,
    ) -> List[Tool]:
        """Return the top-k tools most similar to *query*."""
        reg = registry or self.registry
        if reg is None:
            raise ValueError("ToolRegistry must be provided either at init or in select()")

        k: int = kwargs.get("k", self._default_k)
        tools = reg.all_tools()

        if len(tools) == 0:
            return []

        k = min(k, len(tools))
        model = self._get_model()

        if model is not None:
            descriptions = [f"{t.name}: {t.description}" for t in tools]
            all_texts = [query] + descriptions
            embeddings = model.encode(all_texts, convert_to_numpy=True)

            query_vec: np.ndarray = embeddings[0]
            tool_matrix: np.ndarray = embeddings[1:]
            scores = self._cosine_similarities(query_vec, tool_matrix)
        else:
            scores = self._keyword_similarity_scores(query, tools)

        top_indices = np.argsort(scores)[::-1][:k]
        return [tools[i] for i in top_indices]

    def name(self) -> str:
        return f"retrieval({self._model_name})"
