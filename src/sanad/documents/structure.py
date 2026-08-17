"""Recognising legal document structure in raw text.

Contracts, statutes and pleadings are written as a hierarchy of numbered
provisions. Detecting that hierarchy is what lets Sanad (a) avoid splitting a
clause across two chunks and (b) cite "Article 12.3" instead of only a page.

The detectors are heuristic but conservative: when a line is ambiguous it is
treated as ordinary body text, because a missed heading only costs a little
retrieval quality whereas a false heading corrupts a citation.
"""

from __future__ import annotations

import re

from sanad.utils.text import fold_digits

# "Article 7", "Section 7.2", "Clause IV", "Schedule B", "Annex 2"…
_EN_LABEL = re.compile(
    r"""^\s*
    (?P<kind>ARTICLE|Article|SECTION|Section|CLAUSE|Clause|PART|Part|
             SCHEDULE|Schedule|ANNEX|Annex|APPENDIX|Appendix|EXHIBIT|Exhibit|
             RECITAL|Recital|CHAPTER|Chapter)
    \s+
    (?P<number>\d+(?:\.\d+)*|[IVXLC]+|[A-Z])
    \s*[:.\-–—)]?\s*
    (?P<title>.*)$
    """,
    re.VERBOSE,
)

# "المادة السابعة", "البند ٧-٢", "الفصل الثالث", "الملحق (أ)"
_AR_LABEL = re.compile(
    r"""^\s*
    (?P<kind>المادة|البند|الفصل|الباب|القسم|الملحق|الجدول|التمهيد|المقدمة|الفقرة)
    \s+
    (?P<number>[()ء-ي0-9٠-٩][()ء-ي0-9٠-٩\s./\-]{0,28}?)
    \s*[:.\-–—)]?\s*
    (?P<title>.*)$
    """,
    re.VERBOSE,
)

# A bare outline number opening a provision: "7.", "7.2", "3.1.4 Payment".
_NUMERIC_LABEL = re.compile(r"^\s*(?P<number>\d{1,3}(?:\.\d{1,3}){0,3})[.)]?\s+(?P<title>\S.*)$")

#: Longest a line may be and still be considered a heading.
_MAX_HEADING_CHARS = 110


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .:-–—\t")


def detect_label(line: str) -> str | None:
    """Return a normalised provision label for ``line``, or ``None``.

    ``"Section 7.2 Payment Terms"`` becomes ``"Section 7.2"``: the number is what
    makes a citation checkable, and dropping the title keeps citation strings
    short and stable.
    """
    text = line.strip()
    if not text or len(text) > _MAX_HEADING_CHARS * 3:
        return None

    match = _EN_LABEL.match(text)
    if match:
        kind = match.group("kind").title()
        return f"{kind} {_clean(match.group('number'))}"

    match = _AR_LABEL.match(text)
    if match:
        number = _clean(match.group("number"))
        # Reject a runaway match that swallowed a whole sentence.
        if number and len(number) <= 30:
            return f"{match.group('kind')} {number}"

    match = _NUMERIC_LABEL.match(fold_digits(text))
    if match and len(text) <= _MAX_HEADING_CHARS:
        # A bare outline number would cite as "contract.pdf — 3", which reads as
        # a typo. The section sign makes it unambiguous in both scripts without
        # inventing a word ("Clause", "المادة") the document never used.
        return f"§ {match.group('number')}"

    return None


def is_heading(line: str, *, style_hint: str | None = None) -> bool:
    """Whether ``line`` opens a new provision or section.

    ``style_hint`` is the DOCX paragraph style name when available; a real
    "Heading 2" style is trusted immediately. For PDFs, where no style
    information survives extraction, the decision falls back to typography-free
    heuristics: a numbered provision opener, or a short unpunctuated line in
    title/upper case.
    """
    if style_hint:
        normalised = style_hint.strip().lower()
        if normalised.startswith("heading") or normalised in {"title", "subtitle"}:
            return True

    text = line.strip()
    if not text or len(text) > _MAX_HEADING_CHARS:
        return False
    if detect_label(text) is not None:
        return True

    # A short line that does not end like a sentence and is cased like a title.
    if text.endswith((".", "،", ",", ";", ":", "؛")):
        return False
    words = text.split()
    if not 1 <= len(words) <= 12:
        return False
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    if all(char.isupper() for char in letters if char.isascii()):
        return any(char.isascii() for char in letters)
    return False

