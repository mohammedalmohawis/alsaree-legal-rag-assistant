"""Text normalisation, tokenisation and sentence segmentation.

These helpers are deliberately dependency-free and bilingual: the lexical half
of the retriever and the chunker both rely on them, and both must behave the
same way for Arabic and English input.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat (short vowels), tatweel and the superscript alef. Stripping these is
# what makes "العَقْد" and "العقد" match in the lexical index.
_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")

# Orthographic variants that Arabic legal drafting uses interchangeably.
_ARABIC_FOLDING = {
    "آ": "ا",  # آ -> ا
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "ى": "ي",  # ى -> ي
    "ة": "ه",  # ة -> ه
    "ؤ": "و",  # ؤ -> و
    "ئ": "ي",  # ئ -> ي
}

# Arabic-Indic and extended Arabic-Indic digits, folded to ASCII so that
# "المادة ٧" and "Article 7" tokenise to the same number.
_DIGIT_FOLDING = {
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
}

_TOKEN_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+", re.UNICODE)

# Sentence terminators in both scripts, including the Arabic question mark and
# the Arabic full stop used in some typesetting.
_SENTENCE_END = re.compile(r"(?<=[.!?؟۔])\s+|\n{2,}")

_WHITESPACE = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def fold_digits(text: str) -> str:
    """Map Arabic-Indic digits onto ASCII digits."""
    return text.translate(str.maketrans(_DIGIT_FOLDING))


def normalise_arabic(text: str) -> str:
    """Fold Arabic orthographic variation so equivalent spellings compare equal.

    Applied only for matching and indexing. Displayed text and text sent to the
    model always keeps its original spelling, because legal terms must be
    preserved verbatim.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.translate(str.maketrans(_ARABIC_FOLDING))
    return fold_digits(text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def tokenize(text: str) -> list[str]:
    """Lower-case, script-normalised tokens for lexical scoring.

    Numbers survive intact (``"1,500.00"`` stays one token) because amounts,
    dates and clause numbers are exactly the terms a lexical index must match
    that a dense embedding tends to blur.
    """
    return _TOKEN_RE.findall(normalise_arabic(text).lower())


def split_sentences(text: str) -> list[str]:
    """Split into sentences, falling back to the whole string when unsplittable."""
    parts = [part.strip() for part in _SENTENCE_END.split(text) if part and part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    """Shorten ``text`` to ``limit`` characters on a word boundary where possible."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    pivot = cut.rfind(" ")
    if pivot > limit * 0.6:
        cut = cut[:pivot]
    return cut.rstrip() + suffix
