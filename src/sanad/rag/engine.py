"""The application engine: indexing, grounded Q&A, summary, comparison, extraction.

This is the only object the UI talks to. It owns the search index and the model
client, and every public method returns a typed result the UI can render without
knowing anything about prompts, retrieval or citations.

Whole-document features (summary, comparison, extraction) read a document
linearly rather than retrieving from it, because "what does this contract say
overall" is not a retrieval question. Where a document exceeds the model's input
budget the result reports the truncation explicitly instead of silently
analysing a prefix.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sanad.config import AppSettings
from sanad.documents.library import DocumentLibrary
from sanad.documents.models import ParsedDocument
from sanad.errors import ModelError, RetrievalError
from sanad.llm import prompts
from sanad.llm.gemini import GeminiClient
from sanad.rag.citations import (
    Citation,
    build_source_block,
    is_not_found,
    verify_markers,
)
from sanad.rag.retriever import HybridRetriever, RetrievedChunk
from sanad.rag.store import SearchIndex
from sanad.utils.language import detect_language

#: Characters of a single document sent to the model in one whole-document call.
#: gemini-3.6-flash accepts far more, but a tighter budget keeps latency and
#: cost predictable and keeps the model's attention on the operative text.
MAX_DOCUMENT_CHARS = 60_000

#: Window size for the map step of summarisation.
SUMMARY_SLICE_CHARS = 14_000


@dataclass(frozen=True)
class IndexReport:
    """What happened during indexing, surfaced to the user after processing."""

    documents: int
    chunks: int
    embedded: int

    @property
    def is_empty(self) -> bool:
        return self.chunks == 0


@dataclass
class AnswerResult:
    """A grounded answer plus the sources it actually relied on."""

    question: str
    answer: str
    language: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    grounded: bool = True

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)


@dataclass
class SummaryResult:
    filename: str
    text: str
    language: str
    truncated: bool = False


@dataclass
class ComparisonResult:
    name_a: str
    name_b: str
    text: str
    language: str
    truncated: bool = False


@dataclass
class ExtractionResult:
    filename: str
    fields: dict[str, Any]
    language: str
    truncated: bool = False

    def populated(self) -> list[str]:
        """Field names the document actually supported a value for."""
        return [
            name
            for name in prompts.EXTRACTION_FIELDS
            if self.fields.get(name) not in (None, "", [], {})
        ]


def _slice_text(text: str, size: int) -> list[str]:
    """Split long text into windows, preferring paragraph boundaries."""
    if len(text) <= size:
        return [text] if text.strip() else []

    slices: list[str] = []
    current: list[str] = []
    length = 0
    for paragraph in text.split("\n\n"):
        if length and length + len(paragraph) > size:
            slices.append("\n\n".join(current))
            current, length = [], 0
        current.append(paragraph)
        length += len(paragraph) + 2
    if current:
        slices.append("\n\n".join(current))
    return [item for item in slices if item.strip()]


def _budget(text: str, limit: int) -> tuple[str, bool]:
    """Trim ``text`` to ``limit`` characters, reporting whether it was trimmed."""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


class RagEngine:
    """Indexes a document library and answers questions against it."""

    def __init__(self, client: GeminiClient, settings: AppSettings | None = None) -> None:
        self._client = client
        self._settings = settings or AppSettings()
        self._index = SearchIndex()
        self._retriever = HybridRetriever(self._index, client, self._settings.retrieval)

    # --- indexing -------------------------------------------------------------

    @property
    def index(self) -> SearchIndex:
        return self._index

    @property
    def is_ready(self) -> bool:
        return bool(self._index)

    def build_index(self, library: DocumentLibrary) -> IndexReport:
        """Embed every chunk in ``library`` and rebuild the search index.

        Embedding is batched by the client, so a 400-chunk library costs a
        handful of requests rather than 400.
        """
        chunks = library.chunks()
        if not chunks:
            self._index.clear()
            return IndexReport(documents=len(library), chunks=0, embedded=0)

        vectors = self._client.embed_documents([chunk.text for chunk in chunks])
        self._index.build(chunks, vectors)
        return IndexReport(documents=len(library), chunks=len(chunks), embedded=len(vectors))

    def clear(self) -> None:
        self._index.clear()

    # --- grounded question answering -----------------------------------------

    def answer(
        self,
        question: str,
        *,
        doc_ids: Sequence[str] | None = None,
        history: list[dict] | None = None,
        language: str | None = None,
    ) -> AnswerResult:
        """Answer ``question`` strictly from the indexed documents.

        The response language follows the question, not the interface, unless
        ``language`` overrides it.
        """
        if not self._index:
            raise RetrievalError("No documents have been indexed yet.")

        resolved = language or detect_language(question, self._settings.default_language)
        retrieved = self._retriever.retrieve(question, doc_ids=doc_ids)

        if not retrieved:
            # No passage cleared the filter, so there is nothing to ground an
            # answer in. Return the refusal directly instead of asking the model
            # to invent one from an empty context.
            return AnswerResult(
                question=question,
                answer=prompts.NOT_FOUND_MESSAGE.get(
                    resolved, prompts.NOT_FOUND_MESSAGE["en"]
                ),
                language=resolved,
                grounded=False,
            )

        source_block, citations = build_source_block(retrieved, resolved)
        prompt = prompts.build_answer_prompt(
            question,
            source_block,
            resolved,
            history=prompts.build_history_digest(history),
        )
        raw = self._client.generate(prompt, system_instruction=prompts.GROUNDING_SYSTEM)

        cleaned, used = verify_markers(raw, citations)
        grounded = not is_not_found(cleaned)

        # Fail closed: a substantive answer with no valid citations is
        # unsupported and must not be shown as grounded.
        if grounded and not used:
            cleaned = prompts.NOT_FOUND_MESSAGE.get(
                resolved, prompts.NOT_FOUND_MESSAGE["en"]
            )
            grounded = False

        return AnswerResult(
            question=question,
            answer=cleaned,
            language=resolved,
            citations=used if grounded else [],
            retrieved=retrieved,
            grounded=grounded,
        )

    # --- whole-document features ---------------------------------------------

    def summarize(
        self, parsed: ParsedDocument, *, language: str = "en", focus: str = ""
    ) -> SummaryResult:
        """Produce a structured brief for one document, map-reduce style."""
        text, truncated = _budget(parsed.full_text, MAX_DOCUMENT_CHARS)
        filename = parsed.document.filename
        slices = _slice_text(text, SUMMARY_SLICE_CHARS)
        if not slices:
            raise ModelError("The document contains no text to summarise.")

        partials: list[str] = []
        for piece in slices:
            partial = self._client.generate(
                prompts.build_summary_map_prompt(
                    piece, language, filename=filename, focus=focus
                ),
                system_instruction=prompts.GROUNDING_SYSTEM,
            )
            if "(no substantive content)" not in partial.lower():
                partials.append(partial)

        if not partials:
            raise ModelError("The document produced no substantive summary content.")

        # A single-slice document is already a coherent summary; running the
        # reduce step on it would only add a round trip and risk paraphrasing
        # away detail.
        if len(partials) == 1:
            return SummaryResult(filename, partials[0], language, truncated)

        combined = self._client.generate(
            prompts.build_summary_reduce_prompt(
                "\n\n---\n\n".join(partials), language, filename=filename, focus=focus
            ),
            system_instruction=prompts.GROUNDING_SYSTEM,
        )
        return SummaryResult(filename, combined, language, truncated)

    def compare(
        self, first: ParsedDocument, second: ParsedDocument, *, language: str = "en"
    ) -> ComparisonResult:
        """Compare two documents and report the legally material differences."""
        if first.document.doc_id == second.document.doc_id:
            raise ModelError("Select two different documents to compare.")

        # Halve the budget per side so the pair still fits one request.
        budget = MAX_DOCUMENT_CHARS // 2
        text_a, cut_a = _budget(first.full_text, budget)
        text_b, cut_b = _budget(second.full_text, budget)
        name_a = first.document.filename
        name_b = second.document.filename

        report = self._client.generate(
            prompts.build_comparison_prompt(text_a, text_b, name_a, name_b, language),
            system_instruction=prompts.GROUNDING_SYSTEM,
        )
        return ComparisonResult(name_a, name_b, report, language, cut_a or cut_b)

    def extract(
        self, parsed: ParsedDocument, *, language: str = "en"
    ) -> ExtractionResult:
        """Extract the key contractual facts as a structured record."""
        text, truncated = _budget(parsed.full_text, MAX_DOCUMENT_CHARS)
        filename = parsed.document.filename

        raw = self._client.generate(
            prompts.build_extraction_prompt(text, language, filename=filename),
            system_instruction=prompts.GROUNDING_SYSTEM,
            response_schema=prompts.EXTRACTION_SCHEMA,
        )
        try:
            fields = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ModelError(f"The extraction response was not valid JSON: {error}") from error
        if not isinstance(fields, dict):
            raise ModelError("The extraction response was not a JSON object.")

        return ExtractionResult(filename, fields, language, truncated)
