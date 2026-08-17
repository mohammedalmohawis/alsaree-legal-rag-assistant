"""Document parsing: text extraction, structure detection and page metadata."""

from __future__ import annotations

import pytest

from sanad.documents.loaders import load_document, load_docx, load_pdf
from sanad.errors import EmptyDocumentError, UnsupportedFormatError
from tests.conftest import build_docx, build_pdf


class TestPdfLoading:
    def test_extracts_text_from_every_page(self, employment_pdf: bytes) -> None:
        blocks, page_count = load_pdf("employment.pdf", employment_pdf)

        assert page_count == 2
        combined = " ".join(block.text for block in blocks)
        assert "Sara Al-Otaibi" in combined
        assert "sixty days written notice" in combined

    def test_blocks_carry_one_based_page_numbers(self, employment_pdf: bytes) -> None:
        blocks, _ = load_pdf("employment.pdf", employment_pdf)

        pages = {block.page for block in blocks}
        assert pages == {1, 2}
        # Termination sits on the second page of the fixture.
        termination = next(b for b in blocks if "sixty days" in b.text)
        assert termination.page == 2

    def test_detects_section_headings(self, employment_pdf: bytes) -> None:
        blocks, _ = load_pdf("employment.pdf", employment_pdf)

        labels = {block.label for block in blocks if block.is_heading}
        assert "Section 3" in labels
        assert "Section 4" in labels

    def test_rejects_a_pdf_with_no_text_layer(self) -> None:
        import pymupdf

        document = pymupdf.open()
        document.new_page()  # a blank page, as a scan without OCR would produce
        data = document.tobytes()
        document.close()

        with pytest.raises(EmptyDocumentError):
            load_document("D1", "scan.pdf", data)


class TestDocxLoading:
    def test_reports_no_page_count(self, employment_docx: bytes) -> None:
        _blocks, page_count = load_docx("employment.docx", employment_docx)

        # DOCX has no pagination before rendering, so citations must not claim one.
        assert page_count is None

    def test_uses_word_heading_styles(self, employment_docx: bytes) -> None:
        blocks, _ = load_docx("employment.docx", employment_docx)

        headings = [block for block in blocks if block.is_heading]
        assert any(block.label == "Section 1" for block in headings)
        assert any(block.label == "Section 2" for block in headings)

    def test_keeps_tables_in_document_order(self, employment_docx: bytes) -> None:
        blocks, _ = load_docx("employment.docx", employment_docx)

        table_rows = [block for block in blocks if block.kind == "table"]
        assert any(block.text == "Monthly salary | SAR 18,500" for block in table_rows)
        # The table follows the payment clause in the fixture and must not be
        # relocated to the end of the document.
        texts = [block.text for block in blocks]
        assert texts.index("Section 2. Payment") < texts.index("Monthly salary | SAR 18,500")

    def test_handles_arabic_content(self, arabic_docx: bytes) -> None:
        blocks, _ = load_docx("عقد.docx", arabic_docx)

        combined = " ".join(block.text for block in blocks)
        assert "18,500" in combined
        assert any(block.label == "المادة 1" for block in blocks)


class TestLoadDocument:
    def test_builds_source_metadata(self, employment_pdf: bytes) -> None:
        parsed = load_document("D1", "employment.pdf", employment_pdf)

        assert parsed.document.doc_id == "D1"
        assert parsed.document.file_type == "pdf"
        assert parsed.document.page_count == 2
        assert parsed.document.has_pages is True
        assert parsed.document.char_count > 0

    def test_docx_document_reports_no_pages(self, employment_docx: bytes) -> None:
        parsed = load_document("D2", "employment.docx", employment_docx)

        assert parsed.document.has_pages is False
        assert parsed.document.page_count is None

    def test_rejects_unsupported_extension(self) -> None:
        with pytest.raises(UnsupportedFormatError) as caught:
            load_document("D1", "brief.txt", b"plain text")

        assert caught.value.filename == "brief.txt"

    def test_rejects_an_empty_docx(self) -> None:
        with pytest.raises(EmptyDocumentError):
            load_document("D1", "blank.docx", build_docx([("", None)]))

    def test_corrupt_bytes_raise_a_typed_error(self) -> None:
        from sanad.errors import DocumentReadError

        with pytest.raises(DocumentReadError) as caught:
            load_document("D1", "broken.pdf", b"not really a pdf")

        assert caught.value.filename == "broken.pdf"
        assert caught.value.detail  # the underlying reason is preserved

    def test_full_text_round_trips_in_reading_order(self) -> None:
        data = build_pdf([["First clause here.", "Second clause here."]])
        parsed = load_document("D1", "order.pdf", data)

        assert parsed.full_text.index("First clause") < parsed.full_text.index("Second clause")
