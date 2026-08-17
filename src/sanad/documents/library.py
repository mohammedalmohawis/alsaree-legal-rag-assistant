"""The in-session collection of parsed documents.

The library owns document identity (``D1``, ``D2``, …) and is the single place
the rest of the application asks "which documents are loaded, and what is in
them". It holds no vectors — that is the index's job — which keeps ingestion
testable without touching the network.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from sanad.config import ChunkingSettings
from sanad.documents.chunking import chunk_document
from sanad.documents.loaders import load_document
from sanad.documents.models import Chunk, ParsedDocument
from sanad.errors import DocumentReadError


class DocumentLibrary:
    """An ordered, de-duplicated set of parsed documents for one session."""

    def __init__(self, settings: ChunkingSettings | None = None) -> None:
        self._settings = settings or ChunkingSettings()
        self._documents: dict[str, ParsedDocument] = {}
        self._sequence = 0

    # --- population -----------------------------------------------------------

    def add(self, filename: str, data: bytes) -> ParsedDocument:
        """Parse, chunk and register one file.

        Re-adding a filename replaces the previous version in place, keeping its
        identifier stable so existing citations in the transcript stay meaningful.
        """
        existing = self.find_by_filename(filename)
        doc_id = existing.document.doc_id if existing else self._next_id()

        parsed = load_document(doc_id, filename, data)
        chunk_document(parsed, self._settings)
        self._documents[doc_id] = parsed
        return parsed

    def add_many(
        self, uploads: Iterable[tuple[str, bytes]]
    ) -> tuple[list[ParsedDocument], list[DocumentReadError]]:
        """Add several files, collecting per-file failures instead of aborting.

        One unreadable upload should never discard the documents that parsed
        correctly, so errors are returned alongside the successes.
        """
        added: list[ParsedDocument] = []
        errors: list[DocumentReadError] = []
        for filename, data in uploads:
            try:
                added.append(self.add(filename, data))
            except DocumentReadError as error:
                errors.append(error)
        return added, errors

    def remove(self, doc_id: str) -> bool:
        return self._documents.pop(doc_id, None) is not None

    def clear(self) -> None:
        self._documents.clear()
        self._sequence = 0

    def _next_id(self) -> str:
        self._sequence += 1
        return f"D{self._sequence}"

    # --- access ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._documents)

    def __bool__(self) -> bool:
        return bool(self._documents)

    def __iter__(self) -> Iterator[ParsedDocument]:
        return iter(self._documents.values())

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._documents

    @property
    def documents(self) -> list[ParsedDocument]:
        return list(self._documents.values())

    @property
    def doc_ids(self) -> list[str]:
        return list(self._documents)

    def get(self, doc_id: str) -> ParsedDocument | None:
        return self._documents.get(doc_id)

    def find_by_filename(self, filename: str) -> ParsedDocument | None:
        for parsed in self._documents.values():
            if parsed.document.filename == filename:
                return parsed
        return None

    def chunks(self, doc_ids: Sequence[str] | None = None) -> list[Chunk]:
        """Every chunk, optionally restricted to a subset of documents."""
        selected = self._documents.values()
        if doc_ids is not None:
            wanted = set(doc_ids)
            selected = [
                parsed for parsed in self._documents.values() if parsed.document.doc_id in wanted
            ]
        return [chunk for parsed in selected for chunk in parsed.chunks]

    def label(self, doc_id: str) -> str:
        """Display name for a document identifier, falling back to the id."""
        parsed = self._documents.get(doc_id)
        return parsed.document.filename if parsed else doc_id


    @property
    def total_chunks(self) -> int:
        return sum(len(parsed.chunks) for parsed in self._documents.values())
