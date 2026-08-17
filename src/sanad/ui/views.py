"""Streamlit views.

Presentation only: this module reads from :mod:`sanad.ui.state`, calls the
engine, and renders. It contains no retrieval, prompting or parsing logic, and
no literal user-facing English or Arabic — every string comes from
:mod:`sanad.i18n`.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

import streamlit as st

from sanad.config import AppSettings
from sanad.errors import DocumentReadError, SanadError
from sanad.i18n import count_phrase, field_label, t
from sanad.rag.citations import Citation
from sanad.rag.engine import AnswerResult
from sanad.ui import state
from sanad.ui.theme import brand_block

_MB = 1024 * 1024


# --- shared helpers -----------------------------------------------------------


def error_message(error: Exception, language: str) -> str:
    """Render any exception as a translated, user-facing sentence."""
    if isinstance(error, DocumentReadError):
        return t(error.message_key, language, filename=error.filename, detail=error.detail)
    if isinstance(error, SanadError):
        return t(error.message_key, language, detail=error.detail)
    return t("error_generic", language, detail=str(error))


def render_error(error: Exception, language: str) -> None:
    """Show the actionable message, with the raw detail folded away behind it.

    A lawyer needs to know what to fix; the transport text is still one click
    away for whoever is debugging the deployment.
    """
    message = error_message(error, language)
    st.error(message)

    technical = getattr(error, "technical", "")
    if technical and technical not in message:
        with st.expander(t("technical_detail", language)):
            st.code(technical, language=None)


def _empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""<div class="sanad-empty"><h4>{html.escape(title)}</h4>
        <p>{html.escape(body)}</p></div>""",
        unsafe_allow_html=True,
    )


def render_sources(citations: Sequence[Citation], language: str) -> None:
    """Render the Sources block and a viewer for each cited passage.

    The passage text is shown verbatim so the reader can check the answer
    against the document without leaving the app.
    """
    if not citations:
        return

    rows = "".join(
        f"""<div class="sanad-source">
            <span class="sanad-source-marker">S{citation.number}</span>
            <span>{html.escape(citation.filename)}
            <span class="sanad-source-where">— {html.escape(citation.location(language))}</span></span>
        </div>"""
        for citation in citations
    )
    st.markdown(
        f"""<div class="sanad-sources">
            <div class="sanad-sources-title">{html.escape(t('sources_heading', language))}</div>
            {rows}
        </div>""",
        unsafe_allow_html=True,
    )

    with st.expander(t("show_passage", language)):
        for citation in citations:
            st.markdown(f"**S{citation.number} · {html.escape(citation.render(language))}**")
            # st.text, not st.markdown: contract text is full of "6.", "(a)",
            # "*" and "_", and markdown silently restyles all of them — a
            # numbered clause turns into a renumbered list. The reader is
            # checking this against the source, so it must render verbatim.
            st.text(citation.excerpt)


def _render_answer(message: dict, language: str) -> None:
    st.markdown(message["content"])
    citations: list[Citation] = message.get("citations") or []
    if message.get("ungrounded"):
        st.caption(t("ungrounded_notice", language))
    render_sources(citations, language)


# --- sidebar ------------------------------------------------------------------


def _process_uploads(settings: AppSettings, uploads) -> None:
    """Parse, chunk and index the selected files, reporting per-file failures."""
    language = state.language()
    library = state.library()

    try:
        engine = state.engine(settings)
    except SanadError as error:
        render_error(error, language)
        return

    accepted = []
    for upload in uploads:
        data = upload.getvalue()
        if len(data) > settings.max_upload_mb * _MB:
            st.error(
                t("error_too_large", language, filename=upload.name, size=settings.max_upload_mb)
            )
            continue
        accepted.append((upload.name, data))

    if not accepted:
        return

    with st.spinner(t("processing_parse", language)):
        _added, failures = library.add_many(accepted)

    for failure in failures:
        render_error(failure, language)

    if not library:
        return

    try:
        with st.spinner(t("processing_embed", language)):
            report = engine.build_index(library)
    except SanadError as error:
        render_error(error, language)
        state.mark_stale(True)
        return

    state.set_index_report(report)
    state.mark_stale(False)
    state.clear_results()
    state.cycle_uploader()
    st.rerun()


def render_sidebar(settings: AppSettings) -> None:
    """Workspace controls: language, uploads, document list and search scope."""
    language = state.language()
    library = state.library()

    with st.sidebar:
        st.markdown(brand_block(language, compact=True), unsafe_allow_html=True)

        # A radio always holds a value, so the interface language can never
        # become unset mid-session.
        chosen = st.radio(
            t("language_label", language),
            options=("en", "ar"),
            index=("en", "ar").index(language),
            format_func=lambda code: t(f"language_{code}", language),
            horizontal=True,
            key="sanad_language_choice",
        )
        if chosen != language:
            state.set_language(chosen)
            st.rerun()

        st.divider()
        st.subheader(t("upload_heading", language))
        uploads = st.file_uploader(
            t("upload_label", language),
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help=t("upload_help", language, size=settings.max_upload_mb),
            key=state.uploader_key(),
        )

        label = "reprocess" if library else "process"
        if st.button(t(label, language), use_container_width=True, type="primary"):
            if uploads:
                _process_uploads(settings, uploads)
            elif library:
                _reindex(settings)

        report = state.index_report()
        if report is not None and not report.is_empty:
            st.success(
                t(
                    "processed",
                    language,
                    passages=count_phrase("passages", report.chunks, language),
                    documents=count_phrase("documents", report.documents, language),
                )
            )
        if state.is_stale():
            st.warning(t("index_stale", language))

        st.divider()
        st.subheader(t("document_list", language))
        if not library:
            st.caption(t("no_documents", language))
        else:
            _render_document_list(settings)
            _render_scope_control(settings)
            if st.button(t("clear_all", language), use_container_width=True):
                state.reset_workspace(settings)
                st.rerun()


def _reindex(settings: AppSettings) -> None:
    language = state.language()
    try:
        engine = state.engine(settings)
        with st.spinner(t("processing_embed", language)):
            report = engine.build_index(state.library())
    except SanadError as error:
        render_error(error, language)
        return
    state.set_index_report(report)
    state.mark_stale(False)
    st.rerun()


def _render_document_list(settings: AppSettings) -> None:
    language = state.language()
    library = state.library()

    for parsed in library.documents:
        document = parsed.document
        passages = count_phrase("passages", len(parsed.chunks), language)
        if document.has_pages:
            meta = t(
                "doc_meta_pages",
                language,
                pages=count_phrase("pages", document.page_count, language),
                passages=passages,
            )
        else:
            meta = t("doc_meta_nopages", language, passages=passages)

        row, action = st.columns([5, 1], vertical_alignment="center")
        with row:
            st.markdown(
                f"""<div class="sanad-doc-row">
                    <div class="sanad-doc-name">{html.escape(document.filename)}</div>
                    <div class="sanad-doc-meta">{html.escape(meta)}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with action:
            if st.button(
                "✕",
                key=f"remove_{document.doc_id}",
                help=t("remove_document", language),
            ):
                library.remove(document.doc_id)
                state.set_scope([d for d in state.scope() if d != document.doc_id])
                state.clear_results()
                state.mark_stale(True)
                st.rerun()


def _render_scope_control(settings: AppSettings) -> None:
    """Metadata filter: restrict retrieval to a subset of the library."""
    language = state.language()
    library = state.library()

    selected = st.multiselect(
        t("scope_label", language),
        options=library.doc_ids,
        default=[doc_id for doc_id in state.scope() if doc_id in library],
        format_func=library.label,
        placeholder=t("scope_all", language),
    )
    state.set_scope(selected)


# --- header and footer --------------------------------------------------------


def render_header(language: str) -> None:
    st.markdown(brand_block(language), unsafe_allow_html=True)
    st.markdown(
        f"""<div class="sanad-hero">
            <h1>{html.escape(t('app_title', language))}</h1>
            <p>{html.escape(t('tagline', language))}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def render_footer(language: str) -> None:
    st.divider()
    st.markdown(
        f"""<div class="sanad-disclaimer">
            <strong>{html.escape(t('disclaimer_title', language))}</strong> —
            {html.escape(t('disclaimer', language))}
        </div>""",
        unsafe_allow_html=True,
    )


# --- tabs ---------------------------------------------------------------------


def render_chat_tab(settings: AppSettings) -> None:
    """Grounded question answering with inline citations."""
    language = state.language()
    library = state.library()
    ready = bool(library) and not state.is_stale()

    st.subheader(t("chat_heading", language))

    history = state.chat()
    if not history:
        _empty_state(t("chat_empty_title", language), t("chat_empty_body", language))

    for message in history:
        avatar = "🧑‍⚖️" if message["role"] == "user" else "⚖️"
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                _render_answer(message, language)

    if history and st.button(t("clear_chat", language)):
        state.clear_chat()
        st.rerun()

    if not ready:
        st.info(t("chat_locked", language))

    question = st.chat_input(t("chat_placeholder", language), disabled=not ready)
    if not question:
        return

    state.append_chat("user", question)
    try:
        engine = state.engine(settings)
        with st.spinner(t("answering", language)):
            result: AnswerResult = engine.answer(
                question,
                doc_ids=state.scope() or None,
                history=state.chat()[:-1],
            )
    except SanadError as error:
        state.append_chat("assistant", error_message(error, language))
        st.rerun()
        return

    state.append_chat(
        "assistant",
        result.answer,
        citations=result.citations,
        ungrounded=not result.grounded,
    )
    st.rerun()


def _document_selector(label_key: str, language: str, *, key: str, exclude: str | None = None):
    library = state.library()
    options = [doc_id for doc_id in library.doc_ids if doc_id != exclude]
    if not options:
        return None
    return st.selectbox(
        t(label_key, language), options=options, format_func=library.label, key=key
    )


def render_summary_tab(settings: AppSettings) -> None:
    """Structured, document-grounded brief for one file."""
    language = state.language()
    library = state.library()

    st.subheader(t("summary_heading", language))
    st.caption(t("summary_intro", language))

    if not library:
        _empty_state(t("no_documents", language), t("chat_locked", language))
        return

    doc_id = _document_selector("summary_document", language, key="summary_doc")
    focus = st.text_input(
        t("summary_focus", language), help=t("summary_focus_help", language)
    )

    if st.button(t("summary_run", language), type="primary"):
        parsed = library.get(doc_id) if doc_id else None
        if parsed is not None:
            try:
                engine = state.engine(settings)
                with st.spinner(t("summarizing", language)):
                    state.set_result(
                        "summary",
                        engine.summarize(parsed, language=language, focus=focus),
                    )
            except SanadError as error:
                render_error(error, language)

    summary = state.result("summary")
    if summary is not None:
        st.divider()
        st.markdown(f"#### {html.escape(summary.filename)}")
        if summary.truncated:
            st.warning(t("summary_truncated", language))
        st.markdown(summary.text)


def render_compare_tab(settings: AppSettings) -> None:
    """Side-by-side clause comparison of two documents."""
    language = state.language()
    library = state.library()

    st.subheader(t("compare_heading", language))
    st.caption(t("compare_intro", language))

    if len(library) < 2:
        _empty_state(t("compare_empty_title", language), t("compare_need_two", language))
        return

    left, right = st.columns(2)
    with left:
        first = _document_selector("compare_first", language, key="compare_a")
    with right:
        second = _document_selector("compare_second", language, key="compare_b", exclude=first)

    if st.button(t("compare_run", language), type="primary"):
        if not first or not second or first == second:
            st.warning(t("compare_same", language))
        else:
            parsed_a, parsed_b = library.get(first), library.get(second)
            if parsed_a is not None and parsed_b is not None:
                try:
                    engine = state.engine(settings)
                    with st.spinner(t("comparing", language)):
                        state.set_result(
                            "comparison",
                            engine.compare(parsed_a, parsed_b, language=language),
                        )
                except SanadError as error:
                    render_error(error, language)

    comparison = state.result("comparison")
    if comparison is not None:
        st.divider()
        st.markdown(
            f"#### {html.escape(comparison.name_a)} ↔ {html.escape(comparison.name_b)}"
        )
        if comparison.truncated:
            st.warning(t("compare_truncated", language))
        st.markdown(comparison.text)


def render_facts_tab(settings: AppSettings) -> None:
    """Structured key-fact extraction, rendered field by field."""
    language = state.language()
    library = state.library()

    st.subheader(t("facts_heading", language))
    st.caption(t("facts_intro", language))

    if not library:
        _empty_state(t("no_documents", language), t("chat_locked", language))
        return

    doc_id = _document_selector("summary_document", language, key="facts_doc")

    if st.button(t("facts_run", language), type="primary"):
        parsed = library.get(doc_id) if doc_id else None
        if parsed is not None:
            try:
                engine = state.engine(settings)
                with st.spinner(t("extracting", language)):
                    state.set_result("facts", engine.extract(parsed, language=language))
            except SanadError as error:
                render_error(error, language)

    facts = state.result("facts")
    if facts is None:
        return

    st.divider()
    st.markdown(f"#### {html.escape(facts.filename)}")
    if facts.truncated:
        st.warning(t("summary_truncated", language))

    populated = facts.populated()
    if not populated:
        st.info(t("facts_empty", language))
        return

    for name in populated:
        value = facts.fields.get(name)
        st.markdown(f"**{field_label(name, language)}**")
        if isinstance(value, list):
            for item in value:
                st.markdown(f"- {item}")
        else:
            st.markdown(str(value))


#: Tab order for the main workspace.
TABS = (
    ("tab_chat", render_chat_tab),
    ("tab_summary", render_summary_tab),
    ("tab_compare", render_compare_tab),
    ("tab_facts", render_facts_tab),
)


def render_workspace(settings: AppSettings) -> None:
    """Render the four feature tabs."""
    language = state.language()
    tabs = st.tabs([t(key, language) for key, _ in TABS])
    for tab, (_key, renderer) in zip(tabs, TABS):
        with tab:
            renderer(settings)
