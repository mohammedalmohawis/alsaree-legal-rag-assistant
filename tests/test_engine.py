"""The engine end to end: indexing, grounded answers, summary, compare, extract.

These tests drive the real engine, retriever, index and client, with only the
HTTP layer faked. That makes them integration tests of everything below the UI.
"""

from __future__ import annotations

import json

import pytest

from sanad.documents.library import DocumentLibrary
from sanad.errors import ModelError, RetrievalError
from sanad.llm.gemini import GeminiClient
from sanad.llm.prompts import NOT_FOUND_MESSAGE
from sanad.rag.engine import RagEngine
from tests.conftest import FakeTransport, text_response


@pytest.fixture
def library(app_settings, employment_pdf: bytes) -> DocumentLibrary:
    built = DocumentLibrary(app_settings.chunking)
    built.add("Employment Contract.pdf", employment_pdf)
    return built


def build_engine(app_settings, transport) -> RagEngine:
    return RagEngine(
        GeminiClient(app_settings.gemini, transport=transport, sleeper=lambda _: None),
        app_settings,
    )


class TestIndexing:
    def test_embeds_every_chunk(self, app_settings, library, transport) -> None:
        engine = build_engine(app_settings, transport)

        report = engine.build_index(library)

        assert report.documents == 1
        assert report.chunks == library.total_chunks
        assert report.embedded == report.chunks
        assert engine.is_ready

    def test_an_empty_library_yields_an_empty_report(self, app_settings, transport) -> None:
        engine = build_engine(app_settings, transport)

        report = engine.build_index(DocumentLibrary(app_settings.chunking))

        assert report.is_empty
        assert not engine.is_ready
        assert transport.call_count == 0

    def test_indexing_costs_far_fewer_requests_than_chunks(
        self, app_settings, library, transport
    ) -> None:
        engine = build_engine(app_settings, transport)
        engine.build_index(library)

        batch_calls = len(transport.calls_to(":batchEmbedContents"))
        assert batch_calls < library.total_chunks


class TestAnswering:
    def test_answers_with_verified_citations(self, app_settings, library) -> None:
        transport = FakeTransport(text_reply="Notice is sixty days [S1].")
        engine = build_engine(app_settings, transport)
        engine.build_index(library)

        result = engine.answer("What notice is required to terminate?")

        assert result.grounded is True
        assert result.has_citations
        assert result.citations[0].filename == "Employment Contract.pdf"
        assert result.citations[0].page_start is not None

    def test_strips_a_fabricated_citation(self, app_settings, library) -> None:
        transport = FakeTransport(text_reply="The fee is SAR 99,000 [S42].")
        engine = build_engine(app_settings, transport)
        engine.build_index(library)

        result = engine.answer("What is the fee?")

        assert "[S42]" not in result.answer
        assert result.citations == []

    def test_detects_the_refusal_and_reports_no_sources(self, app_settings, library) -> None:
        transport = FakeTransport(text_reply=NOT_FOUND_MESSAGE["en"])
        engine = build_engine(app_settings, transport)
        engine.build_index(library)

        result = engine.answer("What is the buyer's warranty period?")

        assert result.grounded is False
        assert result.citations == []

    def test_answer_language_follows_the_question(self, app_settings, library) -> None:
        transport = FakeTransport(text_reply="الإشعار ستون يوماً [S1].")
        engine = build_engine(app_settings, transport)
        engine.build_index(library)

        result = engine.answer("ما مدة الإشعار المطلوبة لإنهاء العقد؟")

        assert result.language == "ar"

    def test_explicit_language_overrides_detection(self, app_settings, library) -> None:
        engine = build_engine(app_settings, FakeTransport())
        engine.build_index(library)

        assert engine.answer("What is the term?", language="ar").language == "ar"

    def test_scope_restricts_citations_to_the_selected_document(
        self, app_settings, library, employment_docx: bytes
    ) -> None:
        library.add("Second Contract.docx", employment_docx)
        engine = build_engine(app_settings, FakeTransport(text_reply="Answer [S1]."))
        engine.build_index(library)

        second = library.find_by_filename("Second Contract.docx")
        result = engine.answer("What is the salary?", doc_ids=[second.document.doc_id])

        assert result.citations
        assert all(c.filename == "Second Contract.docx" for c in result.citations)

    def test_the_prompt_carries_the_retrieved_context(self, app_settings, library) -> None:
        transport = FakeTransport(text_reply="Answer [S1].")
        engine = build_engine(app_settings, transport)
        engine.build_index(library)

        engine.answer("What is the notice period?")

        prompt = transport.calls_to(":generateContent")[-1]["json"]["contents"][0]["parts"][0]["text"]
        assert "[S1] Employment Contract.pdf" in prompt
        assert "DOCUMENT CONTEXT" in prompt

    def test_questioning_an_empty_index_raises(self, app_settings, transport) -> None:
        engine = build_engine(app_settings, transport)

        with pytest.raises(RetrievalError):
            engine.answer("anything")

    def test_no_matching_scope_returns_the_refusal_without_calling_the_model(
        self, app_settings, library
    ) -> None:
        transport = FakeTransport()
        engine = build_engine(app_settings, transport)
        engine.build_index(library)
        before = len(transport.calls_to(":generateContent"))

        result = engine.answer("What is the term?", doc_ids=["D999"])

        assert result.grounded is False
        assert result.answer == NOT_FOUND_MESSAGE["en"]
        assert len(transport.calls_to(":generateContent")) == before


class TestSummary:
    def test_returns_the_single_partial_without_a_reduce_step(
        self, app_settings, library
    ) -> None:
        transport = FakeTransport(text_reply="A concise structured brief.")
        engine = build_engine(app_settings, transport)
        parsed = library.documents[0]

        summary = engine.summarize(parsed, language="en")

        assert summary.text == "A concise structured brief."
        assert summary.filename == "Employment Contract.pdf"
        assert summary.truncated is False
        # Short document => one map call, no reduce call.
        assert len(transport.calls_to(":generateContent")) == 1

    def test_passes_the_focus_instruction_through(self, app_settings, library) -> None:
        transport = FakeTransport(text_reply="Brief.")
        engine = build_engine(app_settings, transport)

        engine.summarize(library.documents[0], language="en", focus="termination only")

        prompt = transport.calls_to(":generateContent")[0]["json"]["contents"][0]["parts"][0]["text"]
        assert "termination only" in prompt

    def test_raises_when_nothing_substantive_comes_back(self, app_settings, library) -> None:
        transport = FakeTransport(text_reply="(no substantive content)")
        engine = build_engine(app_settings, transport)

        with pytest.raises(ModelError, match="no substantive summary"):
            engine.summarize(library.documents[0], language="en")


class TestComparison:
    def test_compares_two_documents(self, app_settings, library, employment_docx: bytes) -> None:
        library.add("Revised Contract.docx", employment_docx)
        transport = FakeTransport(text_reply="The notice period differs.")
        engine = build_engine(app_settings, transport)

        first, second = library.documents[0], library.documents[1]
        result = engine.compare(first, second, language="en")

        assert result.name_a == "Employment Contract.pdf"
        assert result.name_b == "Revised Contract.docx"
        assert "notice period differs" in result.text

    def test_both_documents_reach_the_prompt(
        self, app_settings, library, employment_docx: bytes
    ) -> None:
        library.add("Revised Contract.docx", employment_docx)
        transport = FakeTransport(text_reply="Report.")
        engine = build_engine(app_settings, transport)

        engine.compare(library.documents[0], library.documents[1], language="en")

        prompt = transport.calls_to(":generateContent")[0]["json"]["contents"][0]["parts"][0]["text"]
        assert "DOCUMENT A: Employment Contract.pdf" in prompt
        assert "DOCUMENT B: Revised Contract.docx" in prompt

    def test_refuses_to_compare_a_document_with_itself(self, app_settings, library) -> None:
        engine = build_engine(app_settings, FakeTransport())
        parsed = library.documents[0]

        with pytest.raises(ModelError, match="two different documents"):
            engine.compare(parsed, parsed, language="en")


class TestExtraction:
    def test_parses_the_structured_record(self, app_settings, library) -> None:
        payload = {
            "contract_type": "Employment agreement",
            "parties": ["Alpha Trading Company", "Sara Al-Otaibi"],
            "governing_law": "Kingdom of Saudi Arabia",
            "penalties": [],
        }
        transport = FakeTransport([text_response(json.dumps(payload))])
        engine = build_engine(app_settings, transport)

        result = engine.extract(library.documents[0], language="en")

        assert result.fields["parties"] == ["Alpha Trading Company", "Sara Al-Otaibi"]
        assert "contract_type" in result.populated()
        # An empty list means "the document is silent", so it is not rendered.
        assert "penalties" not in result.populated()

    def test_requests_the_json_schema(self, app_settings, library) -> None:
        transport = FakeTransport([text_response('{"contract_type":"NDA","parties":[]}')])
        engine = build_engine(app_settings, transport)

        engine.extract(library.documents[0], language="en")

        config = transport.calls_to(":generateContent")[0]["json"]["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert "contract_type" in config["responseSchema"]["properties"]

    def test_reports_invalid_json_clearly(self, app_settings, library) -> None:
        transport = FakeTransport([text_response("not json at all")])
        engine = build_engine(app_settings, transport)

        with pytest.raises(ModelError, match="not valid JSON"):
            engine.extract(library.documents[0], language="en")

    def test_rejects_a_json_array(self, app_settings, library) -> None:
        transport = FakeTransport([text_response("[1, 2, 3]")])
        engine = build_engine(app_settings, transport)

        with pytest.raises(ModelError, match="not a JSON object"):
            engine.extract(library.documents[0], language="en")
