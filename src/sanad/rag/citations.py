"""Building, parsing and verifying source citations.

Citations are the product's central claim, so they are constructed rather than
generated. The model is only ever allowed to *reference* a marker that Sanad put
into the context; the filename, page range and section attached to that marker
come from the parser and the chunker, never from the model.

That inversion is what makes "do not invent page numbers" enforceable rather
than aspirational: :func:`verify_markers` deletes any marker the model emits
that was not in the supplied context, and a chunk with no known location renders
as an explicit "location not available" instead of a plausible-looking page.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from sanad.documents.models import Chunk
from sanad.llm.prompts import NOT_FOUND_MESSAGE
from sanad.rag.retriever import RetrievedChunk
from sanad.utils.text import truncate

#: Matches the markers the prompt asks for: [S1], [S12].
_MARKER_RE = re.compile(r"\[S(\d+)\]")

#: Longest excerpt kept for the "show the passage" expander in the UI.
_EXCERPT_CHARS = 420


@dataclass(frozen=True)
class Citation:
    """One numbered source, with provenance taken from the document itself."""

    number: int
    chunk_id: str
    doc_id: str
    filename: str
    excerpt: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None

    @property
    def marker(self) -> str:
        return f"[S{self.number}]"

    @property
    def has_location(self) -> bool:
        return self.page_start is not None or bool(self.section)

    def location(self, language: str = "en") -> str:
        """The location half of the citation, or an explicit unavailable notice."""
        page_word = "صفحة" if language == "ar" else "Page"
        pages_word = "الصفحات" if language == "ar" else "Pages"
        parts: list[str] = []
        if self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                parts.append(f"{pages_word} {self.page_start}–{self.page_end}")
            else:
                parts.append(f"{page_word} {self.page_start}")
        if self.section:
            parts.append(self.section)
        if parts:
            return " · ".join(parts)
        return "الموضع غير محدد" if language == "ar" else "location not specified"

    def render(self, language: str = "en") -> str:
        """Full citation line, e.g. ``Employment Contract.pdf — Page 4 · Section 7.2``."""
        return f"{self.filename} — {self.location(language)}"


def citation_from_chunk(number: int, chunk: Chunk) -> Citation:
    return Citation(
        number=number,
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        filename=chunk.filename,
        excerpt=truncate(chunk.text, _EXCERPT_CHARS),
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section=chunk.section,
    )


def build_source_block(
    retrieved: Sequence[RetrievedChunk], language: str = "en"
) -> tuple[str, list[Citation]]:
    """Render retrieved passages as the numbered context block sent to the model.

    Each passage is introduced by its marker and its real location, so the model
    can see — and is expected to reuse — the exact citation Sanad will render.
    """
    citations: list[Citation] = []
    blocks: list[str] = []

    for number, item in enumerate(retrieved, start=1):
        citation = citation_from_chunk(number, item.chunk)
        citations.append(citation)
        blocks.append(
            f"[S{number}] {citation.filename} — {citation.location(language)}\n{item.chunk.text}"
        )

    return "\n\n".join(blocks), citations


def extract_markers(text: str) -> list[int]:
    """Return every source number referenced in ``text``, in order of appearance."""
    seen: list[int] = []
    for match in _MARKER_RE.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def verify_markers(text: str, citations: Sequence[Citation]) -> tuple[str, list[Citation]]:
    """Strip unsupported markers and return the citations actually relied upon.

    A marker outside the supplied range is a fabrication, so it is removed from
    the answer rather than rendered — the user must never see a reference that
    points nowhere.
    """
    valid = {citation.number: citation for citation in citations}

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in valid else ""

    cleaned = _MARKER_RE.sub(replace, text)
    # Tidy the spacing left behind by a removed marker.
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.،,;؛])", r"\1", cleaned).strip()

    used = [valid[number] for number in extract_markers(cleaned)]
    return cleaned, used


def is_not_found(answer: str) -> bool:
    """Whether the model returned the exact "not in the documents" refusal.

    Used to suppress a Sources block under an answer that cites nothing.
    """
    stripped = answer.strip().rstrip(".،")
    return any(
        stripped == message.strip().rstrip(".،") or stripped.startswith(message.strip())
        for message in NOT_FOUND_MESSAGE.values()
    )

