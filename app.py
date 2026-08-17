"""Sanad — bilingual, citation-first legal document intelligence.

Streamlit entry point. This file wires the application together and does nothing
else: configuration, ingestion, retrieval, prompting and rendering all live in
the ``sanad`` package under ``src/``.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run app.py` to work from a plain checkout, without requiring
# `pip install -e .` first.
_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402  (import must follow the sys.path setup)

from sanad.config import BRAND, load_settings  # noqa: E402
from sanad.i18n import t  # noqa: E402
from sanad.ui import state, views  # noqa: E402
from sanad.ui.theme import stylesheet  # noqa: E402


def read_secrets() -> dict:
    """Materialise Streamlit Secrets into a plain dict.

    Accessing ``st.secrets`` raises when no secrets file is present, which is the
    normal case for a local run, so the lookup is contained here and every
    failure mode collapses to "no secrets".
    """
    try:
        return {key: st.secrets[key] for key in st.secrets}
    except Exception:
        return {}


def main() -> None:
    settings = load_settings(secrets=read_secrets())

    st.set_page_config(
        page_title=f"{BRAND['product_en']} · {BRAND['tagline_en']}",
        page_icon="⚖️",
        layout="wide",
        # "auto" keeps the workspace open on a desktop but collapses it on a
        # phone, where an expanded sidebar covers the answer.
        initial_sidebar_state="auto",
    )

    state.initialise(settings)
    language = state.language()
    st.markdown(stylesheet(language), unsafe_allow_html=True)

    views.render_sidebar(settings)
    views.render_header(language)

    # Surfaced once, at the top, rather than on every failed action.
    if not settings.gemini.has_api_key:
        st.error(t("error_api_key_missing", language))

    views.render_workspace(settings)
    views.render_footer(language)


if __name__ == "__main__":
    main()
