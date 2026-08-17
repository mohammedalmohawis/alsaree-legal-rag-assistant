"""Sanad — a bilingual, citation-first legal document intelligence platform.

The package is organised in four layers that can be used independently of the
Streamlit front end:

``sanad.documents``
    Ingestion: file parsing into structural blocks, then legal-aware chunking.
``sanad.llm``
    The Gemini REST client and every prompt template used by the product.
``sanad.rag``
    Vector/lexical indexing, hybrid retrieval, citation handling and the
    high-level engine that answers, summarises, compares and extracts.
``sanad.ui``
    Streamlit presentation only. No retrieval or model logic lives here.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "2.0.0"

