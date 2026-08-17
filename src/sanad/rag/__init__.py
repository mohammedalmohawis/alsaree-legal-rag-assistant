"""Retrieval-augmented generation: indexing, hybrid search, citations, engine."""

from __future__ import annotations

from sanad.rag.citations import (
    Citation,
    build_source_block,
    extract_markers,
    verify_markers,
)
from sanad.rag.engine import (
    AnswerResult,
    ComparisonResult,
    ExtractionResult,
    IndexReport,
    RagEngine,
    SummaryResult,
)
from sanad.rag.retriever import HybridRetriever, RetrievedChunk, reciprocal_rank_fusion
from sanad.rag.store import LexicalIndex, SearchIndex

__all__ = [
    "AnswerResult",
    "Citation",
    "ComparisonResult",
    "ExtractionResult",
    "HybridRetriever",
    "IndexReport",
    "LexicalIndex",
    "RagEngine",
    "RetrievedChunk",
    "SearchIndex",
    "SummaryResult",
    "build_source_block",
    "extract_markers",
    "reciprocal_rank_fusion",
    "verify_markers",
]
