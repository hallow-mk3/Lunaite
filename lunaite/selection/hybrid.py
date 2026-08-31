"""
lunaite.selection.hybrid
========================
HybridSelector — Combines lexical/BM25 token matching with dense vector embeddings
via Reciprocal Rank Fusion (RRF).
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


class HybridSelector(Selector):
    """Select top-*k* tools by fusing dense semantic retrieval with lexical keyword scores.

    Combines dense cosine similarity embeddings with BM25/lexical token overlap using
    Reciprocal Rank Fusion (RRF):
        RRF_score(tool) = alpha * (1 / (60 + dense_rank)) + (1 - alpha) * (1 / (60 + lexical_rank))

    Parameters
    ----------
    registry:
        Optional ToolRegistry instance.
    model_name:
        Sentence transformer model name.
    k:
        Default number of tools to return.
    alpha:
        Weight for dense semantic ranking vs lexical ranking (0.0 to 1.0, default 0.6).
    """

    def __init__(
        self,
        registry_or_model: Optional[Union[ToolRegistry, str]] = None,
        model_name: str = _DEFAULT_MODEL,
        k: int = _DEFAULT_K,
        alpha: float = 0.6,
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

        self._alpha = alpha
        self._model = None
        self._has_sentence_transformers: Optional[bool] = None
        self._lock = threading.Lock()

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
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
        """Return cosine similarities between query_vec and each row of matrix."""
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        return matrix_norms @ query_norm

    @staticmethod
    def _lexical_similarity_scores(query: str, tools: List[Tool]) -> np.ndarray:
        """Compute term overlap scores between query and tool schemas."""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        scores = []
        for t in tools:
            target_text = f"{t.name} {t.description} {' '.join(t.parameters.get('properties', {}).keys())}".lower()
            target_tokens = set(re.findall(r"\w+", target_text))
            intersection = query_tokens.intersection(target_tokens)
            score = len(intersection) / max(1, len(query_tokens))
            scores.append(score)
        return np.array(scores, dtype=float)

    def select(
        self,
        query: str,
        registry: Optional[ToolRegistry] = None,
        **kwargs,
    ) -> List[Tool]:
        """Return the top-k tools via hybrid rank fusion."""
        reg = registry or self.registry
        if reg is None:
            raise ValueError("ToolRegistry must be provided either at init or in select()")

        k: int = kwargs.get("k", self._default_k)
        alpha: float = kwargs.get("alpha", self._alpha)
        tools = reg.all_tools()

        if len(tools) == 0:
            return []

        k = min(k, len(tools))
        model = self._get_model()

        # 1. Lexical ranking
        lex_scores = self._lexical_similarity_scores(query, tools)
        lex_rank_order = np.argsort(lex_scores)[::-1]
        lex_ranks = np.zeros(len(tools))
        for rank, idx in enumerate(lex_rank_order):
            lex_ranks[idx] = rank

        # 2. Dense semantic ranking (if available)
        if model is not None:
            descriptions = [f"{t.name}: {t.description}" for t in tools]
            all_texts = [query] + descriptions
            embeddings = model.encode(all_texts, convert_to_numpy=True)

            query_vec: np.ndarray = embeddings[0]
            tool_matrix: np.ndarray = embeddings[1:]
            dense_scores = self._cosine_similarities(query_vec, tool_matrix)
            dense_rank_order = np.argsort(dense_scores)[::-1]
            dense_ranks = np.zeros(len(tools))
            for rank, idx in enumerate(dense_rank_order):
                dense_ranks[idx] = rank

            # Reciprocal Rank Fusion
            rrf_scores = (alpha / (60.0 + dense_ranks)) + ((1.0 - alpha) / (60.0 + lex_ranks))
        else:
            rrf_scores = 1.0 / (60.0 + lex_ranks)

        top_indices = np.argsort(rrf_scores)[::-1][:k]
        return [tools[i] for i in top_indices]

    def name(self) -> str:
        return f"hybrid({self._model_name}, alpha={self._alpha})"
