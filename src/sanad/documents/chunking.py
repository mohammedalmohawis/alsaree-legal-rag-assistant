"""Structure-aware chunking for legal documents.

Splitting on a fixed character count ignores the document and produces chunks
several pages long, so retrieval can tell you *which pages* discuss termination
but not *which clause*.

This packer works at clause granularity instead. It never splits a paragraph
unless that paragraph alone exceeds the hard ceiling, it starts a new chunk when
a new heading or numbered provision begins, and it carries a sentence-aligned
overlap so a provision that straddles a boundary is still retrievable from
either side.

Every chunk records the pages and the section it came from, which is what makes
a verifiable citation possible.
"""

from __future__ import annotations

from sanad.config import ChunkingSettings
from sanad.documents.models import Chunk, ParsedDocument, TextBlock
from sanad.utils.text import split_sentences, truncate

#: Longest heading text kept as a section label when the heading has no number.
_MAX_SECTION_LABEL = 70


def _section_label(block: TextBlock) -> str | None:
    """The section name a block establishes, if any.

    A parsed provision label ("Section 7.2") is preferred over the heading's
    prose because it is short, stable and independently checkable against the
    source document.
    """
    if block.label:
        return block.label
    if block.is_heading:
        return truncate(block.text, _MAX_SECTION_LABEL, suffix="")
    return None


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split a single over-long block on sentence boundaries.

    Falls back to a hard character split only for text with no sentence
    breaks at all, such as a very long table row or an unpunctuated recital.
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
        # A single sentence longer than the ceiling still has to be cut.
        while len(current) > max_chars:
            pieces.append(current[:max_chars])
            current = current[max_chars:]
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece.strip()]


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """Take the trailing ``overlap_chars`` of a chunk, aligned to a sentence.

    Aligning to a sentence keeps the overlap readable, so the model never sees a
    fragment starting mid-word.
    """
    if overlap_chars <= 0 or not text:
        return ""
    if len(text) <= overlap_chars:
        return text

    tail = text[-overlap_chars:]
    sentences = split_sentences(tail)
    if len(sentences) > 1:
        # Drop the leading partial sentence.
        return " ".join(sentences[1:]).strip()
    pivot = tail.find(" ")
    return tail[pivot + 1 :].strip() if pivot != -1 else tail.strip()


class _ChunkBuilder:
    """Accumulates blocks and emits chunks, tracking pages and section labels."""

    def __init__(self, parsed: ParsedDocument, settings: ChunkingSettings) -> None:
        self._parsed = parsed
        self._settings = settings
        self._chunks: list[Chunk] = []
        self._buffer: list[str] = []
        self._pages: list[int] = []
        self._length = 0
        self._section: str | None = None
        self._pending_overlap = ""
        self._kind = "paragraph"

    @property
    def length(self) -> int:
        return self._length

    def add(self, text: str, page: int | None, kind: str) -> None:
        self._buffer.append(text)
        self._length += len(text) + 2  # account for the joining blank line
        if page is not None:
            self._pages.append(page)
        if kind == "table" and self._kind == "paragraph":
            self._kind = "table"

    def flush(self) -> None:
        """Emit the buffered blocks as a chunk and seed the next overlap."""
        if not self._buffer:
            return

        body = "\n\n".join(self._buffer).strip()
        if not body:
            self._reset()
            return

        text = f"{self._pending_overlap}\n\n{body}".strip() if self._pending_overlap else body
        document = self._parsed.document
        index = len(self._chunks)
        self._chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}-C{index:04d}",
                doc_id=document.doc_id,
                filename=document.filename,
                index=index,
                text=text,
                page_start=min(self._pages) if self._pages else None,
                page_end=max(self._pages) if self._pages else None,
                section=self._section,
                kind=self._kind,
            )
        )
        self._pending_overlap = _overlap_tail(body, self._settings.overlap_chars)
        self._reset()

    def set_section(self, section: str | None) -> None:
        if section:
            self._section = section

    def _reset(self) -> None:
        self._buffer = []
        self._pages = []
        self._length = 0
        self._kind = "paragraph"

    def result(self) -> list[Chunk]:
        return self._chunks


def chunk_blocks(parsed: ParsedDocument, settings: ChunkingSettings) -> list[Chunk]:
    """Pack ``parsed.blocks`` into retrieval chunks.

    The packer flushes on three conditions, in priority order:

    1. A heading arrives and the buffer already holds a usable chunk — this is
       what keeps one clause per chunk.
    2. Adding the next block would pass ``target_chars`` and the buffer is at
       least ``min_chars``.
    3. The buffer reached ``max_chars`` — the hard ceiling.
    """
    builder = _ChunkBuilder(parsed, settings)

    for block in parsed.blocks:
        label = _section_label(block)

        if block.is_heading:
            # Close the previous provision before opening the next one, but only
            # once it is substantial enough to stand alone; otherwise a run of
            # sub-headings would emit a chunk per line.
            if builder.length >= settings.min_chars:
                builder.flush()
            builder.set_section(label)
            builder.add(block.text, block.page, block.kind)
            continue

        builder.set_section(label)

        for piece in _split_oversized(block.text, settings.max_chars):
            would_exceed = builder.length + len(piece) > settings.target_chars
            if would_exceed and builder.length >= settings.min_chars:
                builder.flush()
            builder.add(piece, block.page, block.kind)
            if builder.length >= settings.max_chars:
                builder.flush()

    builder.flush()
    return builder.result()


def chunk_document(parsed: ParsedDocument, settings: ChunkingSettings) -> ParsedDocument:
    """Attach chunks to ``parsed`` in place and return it for chaining."""
    parsed.chunks = chunk_blocks(parsed, settings)
    return parsed

