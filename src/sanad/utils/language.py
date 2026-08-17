"""Per-question language detection.

The answer language is decided independently of the interface language, so an
Arabic question typed while the UI is in English still gets an Arabic answer.
An explicit instruction in the question always wins over script counting.
"""

from __future__ import annotations

import re

from sanad.utils.text import normalise_arabic

_ARABIC_CHARS = re.compile(r"[؀-ۿݐ-ݿ]")
_LATIN_CHARS = re.compile(r"[A-Za-z]")

# Normalised (diacritic-free, alef-folded) forms, because that is what the
# incoming text is reduced to before matching.
_ANSWER_IN_ARABIC = (
    "answer in arabic",
    "respond in arabic",
    "reply in arabic",
    "in arabic please",
    "اجب بالعربيه",
    "اجب بالعربي",
    "الرد بالعربيه",
    "بالعربيه من فضلك",
    "اكتب بالعربيه",
)
_ANSWER_IN_ENGLISH = (
    "answer in english",
    "respond in english",
    "reply in english",
    "in english please",
    "اجب بالانجليزيه",
    "الرد بالانجليزيه",
    "بالانجليزيه من فضلك",
    "اكتب بالانجليزيه",
)


def script_ratio(text: str) -> tuple[int, int]:
    """Return ``(arabic_letters, latin_letters)`` found in ``text``."""
    return len(_ARABIC_CHARS.findall(text)), len(_LATIN_CHARS.findall(text))


def detect_language(text: str, default: str = "en") -> str:
    """Return ``"ar"`` or ``"en"`` for a free-form question.

    Resolution order:

    1. An explicit "answer in <language>" instruction, in either script.
    2. Whichever script contributes more letters.
    3. ``default``, when the text carries no letters at all (e.g. "7.2?").
    """
    if not text or not text.strip():
        return default

    probe = normalise_arabic(text).lower()
    # Checked before script counting so "اشرح البند answer in English" works.
    arabic_request = any(phrase in probe for phrase in _ANSWER_IN_ARABIC)
    english_request = any(phrase in probe for phrase in _ANSWER_IN_ENGLISH)
    if arabic_request and not english_request:
        return "ar"
    if english_request and not arabic_request:
        return "en"

    arabic, latin = script_ratio(text)
    if arabic == 0 and latin == 0:
        return default
    return "ar" if arabic > latin else "en"


def is_rtl(language: str) -> bool:
    """Whether the interface should be laid out right-to-left."""
    return language == "ar"
