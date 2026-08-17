"""Data model for ingested documents.

Three levels, each immutable:

``SourceDocument``
    One uploaded file and its provenance.
``TextBlock``
    A paragraph, heading or table row *as it appeared in the file*, carrying the
    page it came from. Blocks are what the loaders produce.
``Chunk``
    A retrieval unit assembled from consecutive blocks, carrying enough
    provenance to cite it precisely.

Location metadata is ``Optional`` throughout, and that is load-bearing: DOCX
files have no page concept, so a DOCX chunk honestly reports ``page_start is
None`` rather than inventing a page number.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceDocument:
    """An uploaded file after successful parsing."""

    doc_id: str
    filename: str
    file_type: str
    char_count: int
    page_count: int | None = None
    block_count: int = 0

    @property
    def has_pages(self) -> bool:
        """Whether citations for this document can carry a page number."""
        return self.page_count is not None


@dataclass(frozen=True)
class TextBlock:
    """A single structural unit of text extracted from a file."""

    text: str
    page: int | None = None
    is_heading: bool = False
    #: Clause or article label parsed from the text, e.g. ``"Section 7.2"``.
    label: str | None = None
    #: ``"paragraph"``, ``"heading"`` or ``"table"``.
    kind: str = "paragraph"

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("TextBlock requires non-empty text")


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit with full provenance back to its source location."""

    chunk_id: str
    doc_id: str
    filename: str
    index: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    #: Nearest enclosing heading or clause label, used in citations.
    section: str | None = None
    kind: str = "paragraph"

    @property
    def char_count(self) -> int:
        return len(self.text)



@dataclass
class ParsedDocument:
    """A :class:`SourceDocument` together with everything derived from it."""

    document: SourceDocument
    blocks: list[TextBlock] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """The document reassembled in reading order.

        Used by the whole-document features (summary, extraction, comparison),
        which read the document linearly instead of retrieving from it.
        """
        return "\n\n".join(block.text for block in self.blocks)

