"""Hybrid retrieval: dense + lexical fusion, then diversity-aware selection.

Three stages, each solving a specific failure of plain top-k similarity search:

1. **Dual candidate generation.** Dense search retrieves paraphrases; BM25
   retrieves exact clause numbers, dates and amounts. Each contributes its own
   candidate list.
2. **Reciprocal rank fusion.** The two lists are merged by rank, not by score.
   Cosine similarity and BM25 live on incomparable scales, and RRF needs no
   per-corpus weight tuning to combine them.
3. **Maximal marginal relevance.** Contracts are full of near-identical
   boilerplate, and a naive top-6 often returns six copies of the same clause
   from different documents. MMR trades a little relevance for coverage, so the
   context the model sees spans the distinct provisions that matter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sanad.config import RetrievalSettings
from sanad.documents.models import Chunk
from sanad.errors import RetrievalError
from sanad.rag.store import SearchIndex

try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment,misc]


class QueryEmbedder(Protocol):  # pragma: no cover - structural interface only
    """The single method the retriever needs from the model client."""

    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk selected for the answer context, with its provenance in the ranking."""

    chunk: Chunk
    score: float
    rank: int
    dense_rank: int | None = None
    lexical_rank: int | None = None

    @property
    def matched_lexically(self) -> bool:
        return self.lexical_rank is not None

    @property
    def matched_semantically(self) -> bool:
        return self.dense_rank is not None


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[int, float]]], rrf_k: int = 60
) -> dict[int, float]:
    """Fuse ranked lists into one score per position.

    Each list contributes ``1 / (rrf_k + rank)``. A passage that both rankings
    place near the top outranks one that either ranking places first alone,
    which is exactly the behaviour wanted for a legal query mixing a concept
    with a specific figure.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (position, _score) in enumerate(ranking, start=1):
            fused[position] = fused.get(position, 0.0) + 1.0 / (rrf_k + rank)
    return fused


def maximal_marginal_relevance(
    candidates: Sequence[int],
    relevance: dict[int, float],
    vectors: np.ndarray | None,
    top_k: int,
    lambda_multiplier: float,
) -> list[int]:
    """Select ``top_k`` positions balancing relevance against redundancy.

    Falls back to pure relevance ordering when no vectors are available, so the
    retriever still works if the index was built without a dense half.
    """
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda position: -relevance.get(position, 0.0))
    if vectors is None or len(ordered) <= 1 or top_k <= 1:
        return ordered[:top_k]

    # Min-max normalise relevance so it is commensurate with cosine similarity.
    values = [relevance.get(position, 0.0) for position in ordered]
    low, high = min(values), max(values)
    spread = high - low
    normalised = {
        position: (relevance.get(position, 0.0) - low) / spread if spread else 1.0
        for position in ordered
    }

    index_of = {position: offset for offset, position in enumerate(candidates)}
    similarity = vectors @ vectors.T

    selected: list[int] = [ordered[0]]
    remaining = ordered[1:]

    while remaining and len(selected) < top_k:
        best_position = remaining[0]
        best_value = -float("inf")
        for position in remaining:
            redundancy = max(
                float(similarity[index_of[position], index_of[chosen]]) for chosen in selected
            )
            value = (
                lambda_multiplier * normalised[position]
                - (1.0 - lambda_multiplier) * redundancy
            )
            if value > best_value:
                best_value = value
                best_position = position
        selected.append(best_position)
        remaining.remove(best_position)

    return selected


class HybridRetriever:
    """Turns a natural-language question into a ranked, citable context set."""

    def __init__(
        self,
        index: SearchIndex,
        embedder: QueryEmbedder,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._settings = settings or RetrievalSettings()

    def retrieve(
        self,
        query: str,
        *,
        doc_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the passages that should ground an answer to ``query``.

        ``doc_ids`` scopes the search to specific documents; ``None`` searches
        every indexed document.
        """
        if not self._index:
            raise RetrievalError("No documents have been indexed yet.")
        if not query.strip():
            return []

        limit = top_k or self._settings.top_k
        pool = max(self._settings.candidate_pool, limit)
        allowed: set[int] | None = self._index.allowed_positions(doc_ids)
        if allowed is not None and not allowed:
            return []

        query_vector = self._embedder.embed_query(query)
        dense = self._index.dense_search(query_vector, pool, allowed)
        lexical = self._index.lexical_search(query, pool, allowed)
        if not dense and not lexical:
            return []

        fused = reciprocal_rank_fusion([dense, lexical], self._settings.rrf_k)
        dense_ranks = {position: rank for rank, (position, _) in enumerate(dense, start=1)}
        lexical_ranks = {position: rank for rank, (position, _) in enumerate(lexical, start=1)}

        candidates = [
            position
            for position, score in sorted(fused.items(), key=lambda item: -item[1])
            if score > self._settings.min_score
        ][:pool]

        vectors = self._index.vectors_at(candidates)
        chosen = maximal_marginal_relevance(
            candidates, fused, vectors, limit, self._settings.mmr_lambda
        )

        return [
            RetrievedChunk(
                chunk=self._index.chunk_at(position),
                score=round(fused.get(position, 0.0), 6),
                rank=rank,
                dense_rank=dense_ranks.get(position),
                lexical_rank=lexical_ranks.get(position),
            )
            for rank, position in enumerate(chosen, start=1)
        ]
