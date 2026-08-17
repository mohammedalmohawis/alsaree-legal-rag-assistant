"""Every prompt Sanad sends to the model.

Prompts live in one module, take plain strings, and return plain strings. That
keeps them free of any dependency on retrieval or the UI and makes each one
directly assertable in a unit test.

Two rules are repeated in every template because they are the product's core
promise: answer only from the supplied context, and cite the supplied source
markers. A model that cannot satisfy both is instructed to say so.
"""

from __future__ import annotations

from typing import Any

#: The exact refusal text, per language. The UI matches on these strings to
#: suppress a "Sources" block that would otherwise sit under a non-answer.
NOT_FOUND_MESSAGE = {
    "en": "This information was not found in the uploaded documents.",
    "ar": "لم يتم العثور على هذه المعلومات في المستندات المرفوعة.",
}

_LANGUAGE_NAME = {"en": "English", "ar": "Arabic"}


def language_name(language: str) -> str:
    return _LANGUAGE_NAME.get(language, _LANGUAGE_NAME["en"])


GROUNDING_SYSTEM = """
You are the document analyst for Abdulaziz Al-Saree Law Office, working inside \
the Sanad legal document platform.

Absolute rules:
1. Use ONLY the DOCUMENT CONTEXT supplied in the user message. You have no other \
knowledge of these documents and must not supply general legal knowledge, \
precedent, statutes or commentary that is not in the context.
2. Never invent a clause, party, date, amount, obligation or citation. If the \
context does not settle the question, say so plainly.
3. Cite your evidence. Every factual sentence must end with one or more source \
markers exactly as they appear in the context, for example [S1] or [S2][S4]. \
Never write a marker that is not in the context, and never alter a marker.
4. Quote defined legal terms, party names, figures and dates exactly as written \
in the source. Do not translate a defined term when quoting it; you may gloss it.
5. You are an analysis tool, not counsel. Do not advise on a course of action or \
predict an outcome.
""".strip()


def build_answer_prompt(
    question: str,
    source_block: str,
    language: str,
    *,
    history: str = "",
) -> str:
    """Assemble the grounded question-answering prompt.

    ``source_block`` is the numbered, metadata-labelled context produced by
    :func:`sanad.rag.citations.build_source_block`.
    """
    conversation = f"\nEARLIER CONVERSATION (context only, not evidence):\n{history}\n" if history else ""
    return f"""RESPONSE LANGUAGE: {language_name(language)}. Write the entire answer in this language.

DOCUMENT CONTEXT
Each block below is one retrieved passage. The bracketed marker on the first \
line is the citation you must use when you rely on that passage.

{source_block}
{conversation}
QUESTION
{question}

INSTRUCTIONS
- Answer only from the DOCUMENT CONTEXT above.
- End every factual sentence with the marker(s) of the passage that supports it.
- If the context does not contain the answer, reply with exactly this sentence \
and nothing else: {NOT_FOUND_MESSAGE.get(language, NOT_FOUND_MESSAGE['en'])}
- Be specific and concise. Prefer the document's own wording for operative terms.

ANSWER:"""


def build_summary_map_prompt(
    section_text: str,
    language: str,
    *,
    filename: str,
    focus: str = "",
) -> str:
    """Summarise one slice of a document (the map step)."""
    focus_line = f"\nUSER FOCUS: {focus}" if focus.strip() else ""
    return f"""Summarise the following extract from the legal document "{filename}".

RESPONSE LANGUAGE: {language_name(language)}{focus_line}

RULES
- Summarise only what the extract states. Add no legal facts, conclusions or advice.
- Preserve party names, defined terms, dates and amounts exactly.
- If the extract contains no substantive provisions, reply with: (no substantive content)

EXTRACT
{section_text}

SUMMARY:"""


def build_summary_reduce_prompt(
    partial_summaries: str,
    language: str,
    *,
    filename: str,
    focus: str = "",
) -> str:
    """Combine per-slice summaries into one structured brief (the reduce step)."""
    focus_line = f"\nUSER FOCUS: {focus}" if focus.strip() else ""
    return f"""Combine the partial summaries below into one structured brief for the \
legal document "{filename}".

RESPONSE LANGUAGE: {language_name(language)}{focus_line}

RULES
- Use only the partial summaries. Introduce no new facts.
- Merge duplicates; keep every distinct obligation, date and amount.
- Omit a heading entirely if the summaries say nothing about it. Do not write \
"not specified" under a heading just to fill it.

REQUIRED STRUCTURE (use these headings, in the response language)
1. Purpose and parties
2. Key obligations
3. Financial terms
4. Term, renewal and termination
5. Governing law and dispute resolution
6. Points requiring a lawyer's attention

PARTIAL SUMMARIES
{partial_summaries}

STRUCTURED BRIEF:"""


def build_comparison_prompt(
    document_a: str,
    document_b: str,
    name_a: str,
    name_b: str,
    language: str,
) -> str:
    """Compare two contracts clause by clause."""
    return f"""Compare the two legal documents below and report the differences that \
would matter to a lawyer reviewing them.

RESPONSE LANGUAGE: {language_name(language)}

RULES
- Base every statement strictly on the two texts. Never infer an intention.
- Quote the differing wording where a quotation makes the difference clear.
- Refer to the documents by name: "{name_a}" and "{name_b}".
- If a category has no differences, write one line saying there are none. Do not \
manufacture a difference to fill a category.

REQUIRED STRUCTURE (use these headings, in the response language)
1. Parties
2. Clauses present in one document only
3. Changed obligations
4. Changed dates and deadlines
5. Changed amounts and payment terms
6. Changed governing law, jurisdiction or dispute resolution
7. Differences most likely to be material

=== DOCUMENT A: {name_a} ===
{document_a}

=== DOCUMENT B: {name_b} ===
{document_b}

COMPARISON:"""


#: Schema for structured key-fact extraction. Every field is a list of strings so
#: the model can return several parties or deadlines, and an absent fact is an
#: empty list rather than a fabricated placeholder.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "contract_type": {"type": "STRING"},
        "parties": {"type": "ARRAY", "items": {"type": "STRING"}},
        "effective_date": {"type": "STRING"},
        "term": {"type": "STRING"},
        "key_dates": {"type": "ARRAY", "items": {"type": "STRING"}},
        "obligations": {"type": "ARRAY", "items": {"type": "STRING"}},
        "payment_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
        "termination": {"type": "ARRAY", "items": {"type": "STRING"}},
        "renewal": {"type": "ARRAY", "items": {"type": "STRING"}},
        "penalties": {"type": "ARRAY", "items": {"type": "STRING"}},
        "governing_law": {"type": "STRING"},
        "deadlines": {"type": "ARRAY", "items": {"type": "STRING"}},
        "notes": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["contract_type", "parties"],
}

#: Field order used when rendering extraction results in the UI.
EXTRACTION_FIELDS = (
    "contract_type",
    "parties",
    "effective_date",
    "term",
    "key_dates",
    "obligations",
    "payment_terms",
    "termination",
    "renewal",
    "penalties",
    "governing_law",
    "deadlines",
    "notes",
)


def build_extraction_prompt(document_text: str, language: str, *, filename: str) -> str:
    """Ask for the key-fact record as JSON matching :data:`EXTRACTION_SCHEMA`."""
    return f"""Extract the key contractual facts from the legal document "{filename}" \
and return them as JSON matching the provided schema.

RESPONSE LANGUAGE for all field values: {language_name(language)}

RULES
- Extract only what the document states. Never infer or complete a fact.
- Leave a string field as "" and an array field as [] when the document is silent. \
An empty value is correct; a guess is not.
- Keep party names, figures, currencies and dates exactly as written, in their \
original script, even when the response language differs.
- Quote or closely paraphrase the document for obligations and termination rights.

DOCUMENT
{document_text}
"""


def build_history_digest(turns: list | None = None, limit: int = 4) -> str:
    """Render the last few turns as background context for a follow-up question.

    Included so pronouns in a follow-up resolve, and explicitly labelled in the
    prompt as context rather than evidence so it can never be cited.
    """
    if not turns:
        return ""
    recent = turns[-limit:]
    lines = []
    for turn in recent:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if content:
            lines.append(f"{role}: {content[:300]}")
    return "\n".join(lines)
