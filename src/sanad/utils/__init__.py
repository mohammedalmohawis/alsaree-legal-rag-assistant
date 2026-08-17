"""Language-agnostic text helpers shared by ingestion, retrieval and the UI."""

from __future__ import annotations

from sanad.utils.language import detect_language, is_rtl, script_ratio
from sanad.utils.text import (
    collapse_whitespace,
    normalise_arabic,
    split_sentences,
    tokenize,
    truncate,
)

__all__ = [
    "collapse_whitespace",
    "detect_language",
    "is_rtl",
    "normalise_arabic",
    "script_ratio",
    "split_sentences",
    "tokenize",
    "truncate",
]
