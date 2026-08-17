"""Configuration resolution, the document library, and translation parity."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from sanad.config import (
    BRAND,
    SUPPORTED_LANGUAGES,
    ChunkingSettings,
    load_settings,
    resolve_api_key,
)
from sanad.documents.library import DocumentLibrary
from sanad.errors import UnsupportedFormatError
from sanad.i18n import AR, CATALOGUES, EN, count_phrase, field_label, translate
from sanad.llm.prompts import EXTRACTION_FIELDS

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class TestApiKeyResolution:
    def test_prefers_secrets_over_the_environment(self) -> None:
        key = resolve_api_key({"GOOGLE_API_KEY": "from-env"}, {"GOOGLE_API_KEY": "from-secrets"})
        assert key == "from-secrets"

    def test_accepts_the_documented_aliases(self) -> None:
        assert resolve_api_key({"GEMINI_API_KEY": "a"}) == "a"
        assert resolve_api_key({"API_KEY": "b"}) == "b"

    def test_ignores_a_blank_value(self) -> None:
        assert resolve_api_key({"GOOGLE_API_KEY": "   "}) is None

    def test_returns_none_when_unset(self) -> None:
        assert resolve_api_key({}, {}) is None


class TestSettings:
    def test_uses_defaults_when_nothing_is_set(self) -> None:
        settings = load_settings({}, {}, read_dotenv=False)

        assert settings.gemini.chat_model == "gemini-3.6-flash"
        assert settings.gemini.embedding_model == "gemini-embedding-2"
        assert settings.gemini.has_api_key is False
        assert settings.default_language == "en"

    def test_reads_overrides_from_the_environment(self) -> None:
        settings = load_settings(
            {
                "GOOGLE_API_KEY": "k",
                "SANAD_TOP_K": "12",
                "SANAD_CHUNK_TARGET_CHARS": "900",
                "SANAD_DEFAULT_LANGUAGE": "ar",
                "SANAD_TEMPERATURE": "0.4",
            },
            {},
            read_dotenv=False,
        )

        assert settings.retrieval.top_k == 12
        assert settings.chunking.target_chars == 900
        assert settings.default_language == "ar"
        assert settings.gemini.temperature == 0.4

    def test_a_malformed_override_falls_back_to_the_default(self) -> None:
        # A typo in one variable must not stop the office from working.
        settings = load_settings({"SANAD_TOP_K": "not-a-number"}, {}, read_dotenv=False)
        assert settings.retrieval.top_k == 6

    def test_an_unknown_language_falls_back_to_english(self) -> None:
        settings = load_settings({"SANAD_DEFAULT_LANGUAGE": "fr"}, {}, read_dotenv=False)
        assert settings.default_language == "en"

    def test_settings_are_immutable(self) -> None:
        settings = load_settings({}, {}, read_dotenv=False)
        with pytest.raises(FrozenInstanceError):
            settings.retrieval.top_k = 99  # type: ignore[misc]


class TestDocumentLibrary:
    def test_assigns_sequential_identifiers(self, employment_pdf, employment_docx) -> None:
        library = DocumentLibrary(ChunkingSettings())

        first = library.add("a.pdf", employment_pdf)
        second = library.add("b.docx", employment_docx)

        assert first.document.doc_id == "D1"
        assert second.document.doc_id == "D2"
        assert len(library) == 2

    def test_re_adding_a_filename_keeps_its_identifier(self, employment_pdf) -> None:
        library = DocumentLibrary(ChunkingSettings())
        library.add("a.pdf", employment_pdf)

        # A stable id keeps citations already shown in the transcript meaningful.
        again = library.add("a.pdf", employment_pdf)

        assert again.document.doc_id == "D1"
        assert len(library) == 1

    def test_collects_failures_without_losing_good_documents(self, employment_pdf) -> None:
        library = DocumentLibrary(ChunkingSettings())

        added, errors = library.add_many(
            [("good.pdf", employment_pdf), ("bad.txt", b"nope")]
        )

        assert len(added) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], UnsupportedFormatError)
        assert len(library) == 1

    def test_chunks_can_be_scoped_to_documents(self, employment_pdf, employment_docx) -> None:
        library = DocumentLibrary(ChunkingSettings())
        library.add("a.pdf", employment_pdf)
        library.add("b.docx", employment_docx)

        scoped = library.chunks(["D2"])

        assert scoped
        assert {chunk.doc_id for chunk in scoped} == {"D2"}

    def test_removing_a_document_drops_its_chunks(self, employment_pdf) -> None:
        library = DocumentLibrary(ChunkingSettings())
        library.add("a.pdf", employment_pdf)

        assert library.remove("D1") is True
        assert library.total_chunks == 0
        assert library.remove("D1") is False

    def test_label_falls_back_to_the_identifier(self) -> None:
        assert DocumentLibrary().label("D9") == "D9"

    def test_clear_empties_the_library(self, employment_pdf) -> None:
        library = DocumentLibrary(ChunkingSettings())
        library.add("a.pdf", employment_pdf)
        library.clear()

        assert not library
        assert library.doc_ids == []


class TestTranslations:
    def test_the_two_catalogues_cover_the_same_keys(self) -> None:
        missing_in_arabic = set(EN) - set(AR)
        missing_in_english = set(AR) - set(EN)

        assert not missing_in_arabic, f"untranslated: {sorted(missing_in_arabic)}"
        assert not missing_in_english, f"English missing: {sorted(missing_in_english)}"

    def test_no_string_is_empty(self) -> None:
        for language, catalogue in CATALOGUES.items():
            blank = [key for key, value in catalogue.items() if not value.strip()]
            assert not blank, f"{language} has blank entries: {blank}"

    def test_placeholders_match_across_languages(self) -> None:
        # A mismatched placeholder would render as an untranslated template.
        for key, english in EN.items():
            assert set(_PLACEHOLDER.findall(english)) == set(
                _PLACEHOLDER.findall(AR[key])
            ), f"placeholder mismatch in '{key}'"

    def test_interpolates_values(self) -> None:
        assert "42" in translate("processed", "en", passages=42, documents=1)

    def test_missing_values_degrade_to_the_template(self) -> None:
        # Better a visible placeholder than an exception inside a Streamlit rerun.
        assert "{passages}" in translate("processed", "en")

    def test_unknown_language_falls_back_to_english(self) -> None:
        assert translate("tab_chat", "fr") == EN["tab_chat"]

    def test_unknown_key_returns_the_key(self) -> None:
        assert translate("no_such_key", "en") == "no_such_key"

    @pytest.mark.parametrize("field", EXTRACTION_FIELDS)
    def test_every_extraction_field_is_labelled_in_both_languages(self, field: str) -> None:
        for language in SUPPORTED_LANGUAGES:
            label = field_label(field, language)
            assert label != f"field_{field}", f"missing {language} label for {field}"

    def test_the_product_is_named_consistently_in_both_catalogues(self) -> None:
        # Guards against a rebrand landing in one language and not the other.
        for key in ("app_title", "disclaimer"):
            assert BRAND["product_en"] in EN[key], f"English '{key}' does not name the product"
            assert BRAND["product_ar"] in AR[key], f"Arabic '{key}' does not name the product"


class TestCountPhrase:
    def test_uses_the_singular_for_one(self) -> None:
        assert count_phrase("passages", 1, "en") == "1 passage"
        assert count_phrase("documents", 1, "en") == "1 document"
        assert count_phrase("pages", 1, "en") == "1 page"

    def test_uses_the_plural_for_other_counts(self) -> None:
        assert count_phrase("passages", 9, "en") == "9 passages"
        assert count_phrase("documents", 0, "en") == "0 documents"

    def test_agrees_in_arabic(self) -> None:
        assert count_phrase("passages", 1, "ar") == "مقطع واحد"
        assert count_phrase("documents", 3, "ar") == "3 مستندات"

    def test_no_count_noun_is_left_ungrammatical(self) -> None:
        # Guards the "1 passages" defect: the singular form must never carry
        # the raw count placeholder.
        for noun in ("passages", "documents", "pages"):
            for language in ("en", "ar"):
                assert "{count}" not in count_phrase(noun, 1, language)

