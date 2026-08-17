"""Chunking: size discipline, structural boundaries, overlap and metadata."""

from __future__ import annotations

from sanad.config import ChunkingSettings
from sanad.documents.chunking import chunk_blocks
from sanad.documents.models import ParsedDocument, SourceDocument, TextBlock
from sanad.documents.structure import detect_label, is_heading

SETTINGS = ChunkingSettings(target_chars=400, max_chars=700, min_chars=80, overlap_chars=60)


def make_parsed(blocks, doc_id: str = "D1", filename: str = "contract.pdf", pages=2):
    document = SourceDocument(
        doc_id=doc_id,
        filename=filename,
        file_type="pdf",
        char_count=sum(len(block.text) for block in blocks),
        page_count=pages,
        block_count=len(blocks),
    )
    return ParsedDocument(document=document, blocks=list(blocks))


class TestStructureDetection:
    def test_detects_english_provision_labels(self) -> None:
        assert detect_label("Section 7.2 Payment Terms") == "Section 7.2"
        assert detect_label("ARTICLE 12") == "Article 12"
        assert detect_label("Schedule B — Fees") == "Schedule B"

    def test_detects_arabic_provision_labels(self) -> None:
        assert detect_label("المادة 7 الأجر") == "المادة 7"

    def test_preserves_the_source_numerals_in_a_label(self) -> None:
        # A citation must read back exactly as the document is numbered, so
        # Arabic-Indic digits are kept rather than folded to ASCII. Matching
        # still works because the lexical index normalises digits at query time.
        assert detect_label("البند ٣ الإنهاء") == "البند ٣"

    def test_detects_bare_outline_numbers(self) -> None:
        # Rendered with a section sign so the citation cannot read as a stray digit.
        assert detect_label("7.2 Payment terms") == "§ 7.2"
        assert detect_label("3. Definitions") == "§ 3"

    def test_ignores_ordinary_prose(self) -> None:
        assert detect_label("The parties agree as follows.") is None
        assert detect_label("Payment shall be made within 30 days of invoice.") is None

    def test_trusts_word_heading_styles(self) -> None:
        assert is_heading("Confidentiality", style_hint="Heading 2") is True
        assert is_heading("Confidentiality", style_hint="Normal") is False

    def test_treats_a_long_line_as_body_text(self) -> None:
        long_line = "Termination " * 20
        assert is_heading(long_line) is False


class TestChunkGeometry:
    def test_respects_the_hard_ceiling(self) -> None:
        blocks = [TextBlock(text=f"Clause {i}. " + "obligation text. " * 12, page=1) for i in range(12)]
        chunks = chunk_blocks(make_parsed(blocks), SETTINGS)

        assert chunks
        assert all(chunk.char_count <= SETTINGS.max_chars * 1.5 for chunk in chunks)

    def test_splits_a_single_oversized_paragraph(self) -> None:
        giant = " ".join(f"Sentence number {i} of the recital." for i in range(120))
        chunks = chunk_blocks(make_parsed([TextBlock(text=giant, page=1)]), SETTINGS)

        assert len(chunks) > 1
        assert all(chunk.char_count <= SETTINGS.max_chars * 1.5 for chunk in chunks)

    def test_short_document_stays_a_single_chunk(self) -> None:
        blocks = [TextBlock(text="A short mutual non-disclosure clause.", page=1)]
        chunks = chunk_blocks(make_parsed(blocks), SETTINGS)

        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_produces_no_chunks_for_no_blocks(self) -> None:
        assert chunk_blocks(make_parsed([]), SETTINGS) == []


class TestChunkStructure:
    def test_a_heading_starts_a_new_chunk(self) -> None:
        blocks = [
            TextBlock(text="Section 1. Term", page=1, is_heading=True, label="Section 1"),
            TextBlock(text="The term is twenty-four months. " * 6, page=1),
            TextBlock(text="Section 2. Payment", page=1, is_heading=True, label="Section 2"),
            TextBlock(text="The fee is SAR 18,500 per month. " * 6, page=1),
        ]
        chunks = chunk_blocks(make_parsed(blocks), SETTINGS)

        sections = [chunk.section for chunk in chunks]
        assert "Section 1" in sections
        assert "Section 2" in sections
        # The two provisions must not be merged into one chunk.
        payment = next(chunk for chunk in chunks if chunk.section == "Section 2")
        assert "twenty-four months" not in payment.text.split("Section 2")[-1]

    def test_unlabelled_text_inherits_the_enclosing_section(self) -> None:
        blocks = [
            TextBlock(text="Section 5. Confidentiality", page=3, is_heading=True, label="Section 5"),
            TextBlock(text="Each party shall keep the information secret.", page=3),
        ]
        chunks = chunk_blocks(make_parsed(blocks), SETTINGS)

        assert all(chunk.section == "Section 5" for chunk in chunks)

    def test_overlap_carries_context_across_a_boundary(self) -> None:
        blocks = [
            TextBlock(text="First clause sentence. " * 20, page=1),
            TextBlock(text="Second clause sentence. " * 20, page=1),
        ]
        chunks = chunk_blocks(make_parsed(blocks), SETTINGS)

        assert len(chunks) > 1
        # Every chunk after the first opens with text carried from its predecessor.
        assert "First clause sentence" in chunks[1].text

    def test_zero_overlap_is_honoured(self) -> None:
        settings = ChunkingSettings(target_chars=200, max_chars=300, min_chars=50, overlap_chars=0)
        blocks = [TextBlock(text="Alpha sentence. " * 15, page=1), TextBlock(text="Beta sentence. " * 15, page=1)]
        chunks = chunk_blocks(make_parsed(blocks), settings)

        assert len(chunks) > 1
        assert "Alpha" not in chunks[-1].text


class TestChunkMetadata:
    def test_chunks_carry_document_provenance(self) -> None:
        blocks = [TextBlock(text="Clause text here. " * 10, page=4)]
        chunks = chunk_blocks(make_parsed(blocks, doc_id="D7", filename="lease.pdf"), SETTINGS)

        chunk = chunks[0]
        assert chunk.doc_id == "D7"
        assert chunk.filename == "lease.pdf"
        assert chunk.chunk_id.startswith("D7-C")

    def test_page_range_spans_the_blocks_it_contains(self) -> None:
        blocks = [
            TextBlock(text="Ends on page one. " * 4, page=1),
            TextBlock(text="Continues on page two. " * 4, page=2),
        ]
        chunks = chunk_blocks(make_parsed(blocks), ChunkingSettings(target_chars=5000, max_chars=9000, min_chars=80, overlap_chars=0))

        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 2

    def test_docx_chunks_report_no_page(self) -> None:
        blocks = [TextBlock(text="A clause with no page.", page=None)]
        chunks = chunk_blocks(make_parsed(blocks, pages=None), SETTINGS)

        assert chunks[0].page_start is None
        assert chunks[0].page_end is None

    def test_indices_are_sequential(self) -> None:
        blocks = [TextBlock(text=f"Clause {i}. " + "body " * 40, page=1) for i in range(6)]
        chunks = chunk_blocks(make_parsed(blocks), SETTINGS)

        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
