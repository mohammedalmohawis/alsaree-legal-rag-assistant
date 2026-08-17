"""The session-local search index: dense vectors plus a BM25 lexical index.

Sanad indexes each chunk twice.

The *dense* half is a NumPy matrix of unit-length Gemini embeddings, so cosine
similarity is a single matrix-vector product. At the scale this product works
at — a few thousand chunks from a handful of contracts — that beats an
approximate-nearest-neighbour library and adds no native dependency.

The *lexical* half is BM25 over script-normalised tokens. Dense retrieval is
strong on paraphrase but blurs exactly the tokens legal work turns on: clause
numbers, dates, amounts and party names. BM25 matches those literally, and the
retriever fuses the two rankings.

Both indexes share one chunk list, so a position is the same chunk in both.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from sanad.documents.models import Chunk
from sanad.utils.text import tokenize

Scored = list[tuple[int, float]]


class LexicalIndex:
    """BM25 over the chunk corpus.

    Parameters follow the standard defaults: ``k1`` controls term-frequency
    saturation and ``b`` how strongly scores are normalised by chunk length.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        self._average_length = 0.0

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._postings = defaultdict(list)
        self._lengths = []

        for position, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)
            # A chunk's section label is worth matching on too: a query naming
            # "Article 12" should reach the passage filed under Article 12 even
            # when the body text never repeats the number.
            if chunk.section:
                tokens += tokenize(chunk.section)
            frequencies: dict[str, int] = defaultdict(int)
            for token in tokens:
                frequencies[token] += 1
            for token, count in frequencies.items():
                self._postings[token].append((position, count))
            self._lengths.append(len(tokens))

        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

    def search(
        self, query: str, limit: int, allowed: set[int] | None = None
    ) -> Scored:
        """Return the ``limit`` highest-scoring positions for ``query``."""
        if not self._lengths:
            return []

        total = len(self._lengths)
        scores: dict[int, float] = defaultdict(float)

        for token in set(tokenize(query)):
            postings = self._postings.get(token)
            if not postings:
                continue
            # Probabilistic IDF, in the +0.5 smoothed form that stays positive.
            idf = math.log(1.0 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, frequency in postings:
                if allowed is not None and position not in allowed:
                    continue
                length_norm = (
                    1.0 - self.b + self.b * (self._lengths[position] / self._average_length)
                    if self._average_length
                    else 1.0
                )
                scores[position] += idf * (
                    frequency * (self.k1 + 1.0) / (frequency + self.k1 * length_norm)
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:limit]


class SearchIndex:
    """Chunks plus their dense and lexical representations."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._lexical = LexicalIndex()
        self._positions_by_doc: dict[str, list[int]] = defaultdict(list)

    # --- construction ---------------------------------------------------------

    def build(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Replace the index contents.

        ``vectors[i]`` must be the embedding of ``chunks[i]``; a length mismatch
        is rejected loudly, because a silent offset would misattribute every
        citation the product produces.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"index needs one vector per chunk: {len(chunks)} chunks, {len(vectors)} vectors"
            )

        self._chunks = list(chunks)
        self._positions_by_doc = defaultdict(list)
        for position, chunk in enumerate(self._chunks):
            self._positions_by_doc[chunk.doc_id].append(position)

        self._matrix = (
            np.asarray(vectors, dtype=np.float32) if vectors else None
        )
        self._lexical.build(self._chunks)

    def clear(self) -> None:
        self._chunks = []
        self._matrix = None
        self._lexical = LexicalIndex()
        self._positions_by_doc = defaultdict(list)

    # --- access ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    def __bool__(self) -> bool:
        return bool(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def chunk_at(self, position: int) -> Chunk:
        return self._chunks[position]

    def vectors_at(self, positions: Sequence[int]) -> np.ndarray | None:
        if self._matrix is None or not positions:
            return None
        return self._matrix[list(positions)]

    def allowed_positions(self, doc_ids: Sequence[str] | None) -> set[int] | None:
        """Positions belonging to ``doc_ids``, or ``None`` to search everything.

        This is the metadata filter: it lets the user scope a question to one
        contract without rebuilding the index.
        """
        if doc_ids is None:
            return None
        allowed: set[int] = set()
        for doc_id in doc_ids:
            allowed.update(self._positions_by_doc.get(doc_id, ()))
        return allowed

    # --- search ---------------------------------------------------------------

    def dense_search(
        self,
        query_vector: Sequence[float],
        limit: int,
        allowed: set[int] | None = None,
    ) -> Scored:
        """Cosine similarity search. Vectors are unit length, so this is a dot product."""
        if self._matrix is None or not len(self._chunks):
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"query has {query.shape[0]} dimensions, index has {self._matrix.shape[1]}"
            )

        similarities = self._matrix @ query
        if allowed is not None:
            mask = np.full(similarities.shape, False)
            if allowed:
                mask[list(allowed)] = True
            similarities = np.where(mask, similarities, -np.inf)

        count = min(limit, len(similarities))
        if count <= 0:
            return []
        # argpartition finds the top-k without sorting the whole corpus.
        top = np.argpartition(-similarities, count - 1)[:count]
        ordered = top[np.argsort(-similarities[top])]
        return [
            (int(position), float(similarities[position]))
            for position in ordered
            if np.isfinite(similarities[position])
        ]

    def lexical_search(
        self, query: str, limit: int, allowed: set[int] | None = None
    ) -> Scored:
        return self._lexical.search(query, limit, allowed)
