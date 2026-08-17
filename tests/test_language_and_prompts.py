"""Language detection, text normalisation and prompt construction."""

from __future__ import annotations

import pytest

from sanad.llm import prompts
from sanad.utils.language import detect_language, is_rtl, script_ratio
from sanad.utils.text import (
    collapse_whitespace,
    normalise_arabic,
    split_sentences,
    tokenize,
    truncate,
)


class TestLanguageDetection:
    def test_detects_english(self) -> None:
        assert detect_language("What is the termination date?") == "en"

    def test_detects_arabic(self) -> None:
        assert detect_language("ما تاريخ انتهاء العقد؟") == "ar"

    def test_uses_the_dominant_script_in_mixed_text(self) -> None:
        assert detect_language("العقد contract عربي") == "ar"
        assert detect_language("Explain the العقد clause in detail please") == "en"

    def test_an_explicit_instruction_beats_the_script(self) -> None:
        assert detect_language("اشرح هذا البند answer in English") == "en"
        assert detect_language("Explain this agreement أجب بالعربية") == "ar"

    def test_handles_diacritics_in_the_instruction(self) -> None:
        assert detect_language("Explain this clause أَجِب بِالعَرَبِيَّة") == "ar"

    def test_falls_back_to_the_default_without_letters(self) -> None:
        assert detect_language("7.2 ?", default="ar") == "ar"
        assert detect_language("") == "en"

    def test_reports_script_counts(self) -> None:
        arabic, latin = script_ratio("عقد contract")
        assert arabic == 3 and latin == 8

    def test_rtl_only_for_arabic(self) -> None:
        assert is_rtl("ar") is True
        assert is_rtl("en") is False


class TestTextNormalisation:
    def test_strips_arabic_diacritics(self) -> None:
        assert normalise_arabic("العَقْد") == normalise_arabic("العقد")

    def test_folds_alef_variants(self) -> None:
        assert normalise_arabic("إجراء") == normalise_arabic("اجراء")

    def test_folds_arabic_indic_digits(self) -> None:
        # Ta marbuta is folded too, so the comparison is against the fully
        # normalised form rather than the original spelling.
        assert normalise_arabic("المادة ٧") == normalise_arabic("المادة 7")
        assert normalise_arabic("المادة ٧").endswith(" 7")

    def test_folds_ta_marbuta(self) -> None:
        assert normalise_arabic("مادة") == normalise_arabic("ماده")

    def test_tokenizer_keeps_numbers_intact(self) -> None:
        # Amounts and clause numbers are exactly what lexical search must match.
        assert "18,500" in tokenize("A salary of SAR 18,500 per month")
        assert "7.2" in tokenize("See Section 7.2 below")

    def test_tokenizer_lowercases_and_drops_punctuation(self) -> None:
        assert tokenize("Termination; Notice.") == ["termination", "notice"]

    def test_collapses_whitespace_but_keeps_paragraphs(self) -> None:
        assert collapse_whitespace("a  b\n\n\n\nc") == "a b\n\nc"

    def test_splits_sentences_in_both_scripts(self) -> None:
        assert len(split_sentences("First one. Second one.")) == 2
        assert len(split_sentences("ما هذا؟ نعم.")) == 2

    def test_split_sentences_handles_unpunctuated_text(self) -> None:
        assert split_sentences("no terminator here") == ["no terminator here"]

    def test_truncate_respects_the_limit(self) -> None:
        assert len(truncate("word " * 100, 40)) <= 41
        assert truncate("short", 40) == "short"
        assert truncate("anything", 0) == ""


class TestAnswerPrompt:
    def test_embeds_question_context_and_language(self) -> None:
        prompt = prompts.build_answer_prompt(
            "What is the notice period?", "[S1] file.pdf — Page 2\nSixty days notice.", "en"
        )

        assert "What is the notice period?" in prompt
        assert "[S1] file.pdf — Page 2" in prompt
        assert "RESPONSE LANGUAGE: English" in prompt

    def test_requests_arabic_when_asked(self) -> None:
        prompt = prompts.build_answer_prompt("سؤال", "[S1] ملف\nنص", "ar")

        assert "RESPONSE LANGUAGE: Arabic" in prompt
        assert prompts.NOT_FOUND_MESSAGE["ar"] in prompt

    def test_instructs_the_model_to_cite(self) -> None:
        prompt = prompts.build_answer_prompt("q", "[S1] f\nt", "en")
        assert "marker" in prompt.lower()

    def test_includes_the_exact_refusal_sentence(self) -> None:
        prompt = prompts.build_answer_prompt("q", "[S1] f\nt", "en")
        assert prompts.NOT_FOUND_MESSAGE["en"] in prompt

    def test_history_is_labelled_as_context_not_evidence(self) -> None:
        prompt = prompts.build_answer_prompt("q", "[S1] f\nt", "en", history="User: earlier")

        assert "not evidence" in prompt
        assert "User: earlier" in prompt

    def test_history_is_omitted_when_absent(self) -> None:
        assert "EARLIER CONVERSATION" not in prompts.build_answer_prompt("q", "s", "en")


class TestSystemInstruction:
    def test_forbids_outside_knowledge_and_invention(self) -> None:
        system = prompts.GROUNDING_SYSTEM

        assert "ONLY the DOCUMENT CONTEXT" in system
        assert "Never invent" in system
        assert "Cite your evidence" in system


class TestHistoryDigest:
    def test_renders_recent_turns(self) -> None:
        digest = prompts.build_history_digest(
            [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}]
        )

        assert "User: first" in digest
        assert "Assistant: second" in digest

    def test_limits_how_far_back_it_reaches(self) -> None:
        turns = [{"role": "user", "content": f"turn {i}"} for i in range(10)]

        digest = prompts.build_history_digest(turns, limit=2)

        assert "turn 9" in digest
        assert "turn 0" not in digest

    def test_empty_history_renders_nothing(self) -> None:
        assert prompts.build_history_digest([]) == ""
        assert prompts.build_history_digest(None) == ""


class TestFeaturePrompts:
    def test_summary_prompt_forbids_added_facts(self) -> None:
        prompt = prompts.build_summary_map_prompt("text", "en", filename="a.pdf")

        assert "Add no legal facts" in prompt
        assert "a.pdf" in prompt

    def test_reduce_prompt_lists_the_required_structure(self) -> None:
        prompt = prompts.build_summary_reduce_prompt("partials", "en", filename="a.pdf")

        for heading in ("Purpose and parties", "Term, renewal and termination"):
            assert heading in prompt

    def test_comparison_prompt_names_both_documents(self) -> None:
        prompt = prompts.build_comparison_prompt("A text", "B text", "old.pdf", "new.pdf", "en")

        assert "old.pdf" in prompt and "new.pdf" in prompt
        assert "Changed amounts and payment terms" in prompt

    def test_comparison_prompt_forbids_manufactured_differences(self) -> None:
        prompt = prompts.build_comparison_prompt("a", "b", "x", "y", "en")
        assert "Do not \nmanufacture" in prompt or "manufacture a difference" in prompt

    def test_extraction_prompt_forbids_guessing(self) -> None:
        prompt = prompts.build_extraction_prompt("doc", "en", filename="a.pdf")

        assert "Never infer" in prompt
        assert "a guess is not" in prompt

    @pytest.mark.parametrize("field", prompts.EXTRACTION_FIELDS)
    def test_every_rendered_field_exists_in_the_schema(self, field: str) -> None:
        assert field in prompts.EXTRACTION_SCHEMA["properties"]

    def test_language_name_falls_back_to_english(self) -> None:
        assert prompts.language_name("de") == "English"
