"""Streamlit session state, kept behind a typed accessor.

Streamlit reruns the whole script on every interaction, so anything that must
survive an interaction lives in ``st.session_state``. Centralising the keys here
keeps that surface small and stops the view functions from inventing new ones.

The engine and the document library are session-scoped by design: uploaded
contracts and their embeddings are never written to disk, so closing the tab
disposes of them.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from sanad.config import AppSettings
from sanad.documents.library import DocumentLibrary
from sanad.errors import ConfigurationError
from sanad.llm.gemini import GeminiClient
from sanad.rag.engine import RagEngine

# Session keys, declared once.
LANGUAGE = "sanad_language"
LIBRARY = "sanad_library"
ENGINE = "sanad_engine"
CHAT = "sanad_chat"
INDEX_REPORT = "sanad_index_report"
INDEX_STALE = "sanad_index_stale"
SCOPE = "sanad_scope"
UPLOADER = "sanad_uploader_generation"
RESULTS = "sanad_results"


def initialise(settings: AppSettings) -> None:
    """Populate any session key that does not exist yet."""
    defaults: dict[str, Any] = {
        LANGUAGE: settings.default_language,
        LIBRARY: DocumentLibrary(settings.chunking),
        CHAT: [],
        INDEX_REPORT: None,
        INDEX_STALE: False,
        SCOPE: [],
        UPLOADER: 0,
        RESULTS: {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- simple accessors ---------------------------------------------------------


def language() -> str:
    return st.session_state.get(LANGUAGE, "en")


def set_language(value: str) -> None:
    st.session_state[LANGUAGE] = value


def library() -> DocumentLibrary:
    return st.session_state[LIBRARY]


def chat() -> list[dict]:
    return st.session_state[CHAT]


def append_chat(role: str, content: str, **extra: Any) -> None:
    st.session_state[CHAT].append({"role": role, "content": content, **extra})


def clear_chat() -> None:
    st.session_state[CHAT] = []


def scope() -> list[str]:
    """Document ids the user restricted the search to; empty means all."""
    return st.session_state.get(SCOPE, [])


def set_scope(doc_ids: list[str]) -> None:
    st.session_state[SCOPE] = doc_ids


def mark_stale(value: bool = True) -> None:
    """Flag that the library changed and the index no longer reflects it."""
    st.session_state[INDEX_STALE] = value


def is_stale() -> bool:
    return bool(st.session_state.get(INDEX_STALE))


def index_report():
    return st.session_state.get(INDEX_REPORT)


def set_index_report(report) -> None:
    st.session_state[INDEX_REPORT] = report


def uploader_key() -> str:
    """A changing widget key, used to reset the file uploader after processing."""
    return f"sanad_upload_{st.session_state.get(UPLOADER, 0)}"


def cycle_uploader() -> None:
    st.session_state[UPLOADER] = st.session_state.get(UPLOADER, 0) + 1


def result(name: str) -> Any | None:
    """Read a cached feature result (summary, comparison, extraction)."""
    return st.session_state.get(RESULTS, {}).get(name)


def set_result(name: str, value: Any) -> None:
    st.session_state.setdefault(RESULTS, {})[name] = value


def clear_results() -> None:
    st.session_state[RESULTS] = {}


# --- engine -------------------------------------------------------------------


def engine(settings: AppSettings) -> RagEngine:
    """Return the session's engine, building it on first use.

    Raises :class:`ConfigurationError` when no API key is configured, which the
    caller renders as a translated message rather than a traceback.
    """
    existing = st.session_state.get(ENGINE)
    if existing is not None:
        return existing

    if not settings.gemini.has_api_key:
        raise ConfigurationError("GOOGLE_API_KEY is not set.")

    created = RagEngine(GeminiClient(settings.gemini), settings)
    st.session_state[ENGINE] = created
    return created


def reset_workspace(settings: AppSettings) -> None:
    """Discard documents, index, conversation and cached results."""
    library().clear()
    existing = st.session_state.get(ENGINE)
    if existing is not None:
        existing.clear()
    clear_chat()
    clear_results()
    set_index_report(None)
    mark_stale(False)
    set_scope([])
    cycle_uploader()
