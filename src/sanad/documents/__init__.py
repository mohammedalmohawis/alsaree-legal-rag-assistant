"""Document ingestion: parsing files into blocks, then into citable chunks."""

from __future__ import annotations

from sanad.documents.chunking import chunk_blocks, chunk_document
from sanad.documents.library import DocumentLibrary
from sanad.documents.loaders import SUPPORTED_EXTENSIONS, load_document, load_docx, load_pdf
from sanad.documents.models import Chunk, ParsedDocument, SourceDocument, TextBlock
from sanad.documents.structure import detect_label, is_heading

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "Chunk",
    "DocumentLibrary",
    "ParsedDocument",
    "SourceDocument",
    "TextBlock",
    "chunk_blocks",
    "chunk_document",
    "detect_label",
    "is_heading",
    "load_docx",
    "load_document",
    "load_pdf",
]
