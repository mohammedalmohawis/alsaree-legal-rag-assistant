"""Citations: construction, rendering, and rejection of fabricated markers.

These are the tests that enforce the product's central promise — a citation is
built from parser metadata, never from model output, and a marker the model
invents is deleted rather than displayed.
"""

from __future__ import annotations

from sanad.documents.models import Chunk
from sanad.rag.citations import (
    build_source_block,
    citation_from_chunk,
    extract_markers,
    is_not_found,
    verify_markers,
)
from sanad.rag.retriever import RetrievedChunk


def chunk(
    index: int = 0,
    *,
    filename: str = "Employment Contract.pdf",
    page_start=4,
    page_end=4,
    section="Section 7.2",
    text: str = "Either party may terminate on sixty days notice.",
) -> Chunk:
    return Chunk(
        chunk_id=f"D1-C{index:04d}",
        doc_id="D1",
        filename=filename,
        index=index,
        text=text,
        page_start=page_start,
        page_end=page_end,
        section=section,
    )


def retrieved(*chunks) -> list:
    return [
        RetrievedChunk(chunk=item, score=1.0 / (rank + 1), rank=rank)
        for rank, item in enumerate(chunks, start=1)
    ]


class TestCitationRendering:
    def test_renders_filename_page_and_section(self) -> None:
        citation = citation_from_chunk(1, chunk())

        assert citation.marker == "[S1]"
        assert citation.render("en") == "Employment Contract.pdf — Page 4 · Section 7.2"

    def test_renders_a_page_range(self) -> None:
        citation = citation_from_chunk(1, chunk(page_start=4, page_end=6))
        assert "Pages 4–6" in citation.render("en")

    def test_renders_in_arabic(self) -> None:
        citation = citation_from_chunk(2, chunk(section="المادة 7"))
        rendered = citation.render("ar")

        assert "صفحة 4" in rendered
        assert "المادة 7" in rendered

    def test_states_plainly_when_no_location_is_known(self) -> None:
        # A DOCX chunk outside any numbered provision: nothing to cite but the file.
        citation = citation_from_chunk(1, chunk(page_start=None, page_end=None, section=None))

        assert citation.has_location is False
        assert "location not specified" in citation.render("en")
        assert "غير محدد" in citation.render("ar")

    def test_falls_back_to_the_section_when_pages_are_absent(self) -> None:
        citation = citation_from_chunk(1, chunk(page_start=None, page_end=None, section="Clause 12"))

        assert citation.has_location is True
        assert citation.render("en") == "Employment Contract.pdf — Clause 12"

    def test_excerpt_is_truncated_for_display(self) -> None:
        citation = citation_from_chunk(1, chunk(text="word " * 400))
        assert len(citation.excerpt) < 500


class TestSourceBlock:
    def test_numbers_sources_from_one(self) -> None:
        block, citations = build_source_block(retrieved(chunk(0), chunk(1, page_start=9, page_end=9)))

        assert [citation.number for citation in citations] == [1, 2]
        assert block.startswith("[S1] Employment Contract.pdf — Page 4 · Section 7.2")
        assert "[S2]" in block

    def test_block_contains_the_full_passage_text(self) -> None:
        block, _ = build_source_block(retrieved(chunk(text="Unique clause wording here.")))
        assert "Unique clause wording here." in block

    def test_empty_retrieval_produces_an_empty_block(self) -> None:
        block, citations = build_source_block([])
        assert block == ""
        assert citations == []


class TestMarkerExtraction:
    def test_finds_markers_in_order_without_duplicates(self) -> None:
        assert extract_markers("A [S2] then B [S1] then C [S2].") == [2, 1]

    def test_returns_nothing_for_an_uncited_answer(self) -> None:
        assert extract_markers("No markers at all.") == []


class TestMarkerVerification:
    def test_keeps_valid_markers_and_reports_them(self) -> None:
        _block, citations = build_source_block(retrieved(chunk(0), chunk(1)))

        cleaned, used = verify_markers("Notice is sixty days [S1].", citations)

        assert "[S1]" in cleaned
        assert [citation.number for citation in used] == [1]

    def test_deletes_a_marker_the_model_invented(self) -> None:
        _block, citations = build_source_block(retrieved(chunk(0)))

        # Only [S1] was supplied; [S7] points at nothing and must not survive.
        cleaned, used = verify_markers("Fee is SAR 100 [S7].", citations)

        assert "[S7]" not in cleaned
        assert used == []
        assert cleaned == "Fee is SAR 100."

    def test_keeps_the_valid_half_of_a_mixed_answer(self) -> None:
        _block, citations = build_source_block(retrieved(chunk(0), chunk(1)))

        cleaned, used = verify_markers("First [S1]. Second [S9]. Third [S2].", citations)

        assert "[S1]" in cleaned and "[S2]" in cleaned and "[S9]" not in cleaned
        assert [citation.number for citation in used] == [1, 2]

    def test_reports_only_the_sources_actually_cited(self) -> None:
        _block, citations = build_source_block(retrieved(chunk(0), chunk(1), chunk(2)))

        _cleaned, used = verify_markers("Only the second one matters [S2].", citations)

        assert [citation.number for citation in used] == [2]


class TestNotFoundDetection:
    def test_recognises_the_english_refusal(self) -> None:
        assert is_not_found("This information was not found in the uploaded documents.")

    def test_recognises_the_arabic_refusal(self) -> None:
        assert is_not_found("لم يتم العثور على هذه المعلومات في المستندات المرفوعة.")

    def test_a_real_answer_is_not_a_refusal(self) -> None:
        assert not is_not_found("The notice period is sixty days [S1].")
