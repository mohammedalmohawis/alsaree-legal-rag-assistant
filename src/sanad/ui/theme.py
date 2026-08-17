"""Sanad's visual identity: palette, typography, layout and direction.

Two deliberate constraints shape this module.

*No web fonts.* Type is a system stack that renders both scripts well on macOS,
Windows and Linux. A law office should not need an outbound request to Google
Fonts to render its own document tool, and an offline install should look right.

*No label rewriting.* The stylesheet handles direction, colour and spacing only.
It never uses ``content:`` to overwrite a widget's text: that technique breaks
silently whenever Streamlit changes a DOM attribute. All translated copy is
supplied by the application through :mod:`sanad.i18n`.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

from sanad.config import BRAND, PALETTE
from sanad.utils.language import is_rtl

_ASSETS = Path(__file__).parent / "assets"

# Latin first, then Arabic faces that ship with the major desktop platforms.
_FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", '
    '"Noto Kufi Arabic", "Noto Naskh Arabic", "Geeza Pro", "Dubai", Tahoma, sans-serif'
)


@lru_cache(maxsize=4)
def logo_data_uri() -> str:
    """The Sanad mark as an inline data URI.

    Inlined so the app has no static-file server dependency and the logo
    survives being embedded in a Streamlit markdown block.
    """
    svg = (_ASSETS / "sanad-mark.svg").read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")


def stylesheet(language: str) -> str:
    """Return the full stylesheet for the given interface language."""
    direction = "rtl" if is_rtl(language) else "ltr"

    return f"""
    <style>
    :root {{
        --sanad-ink: {PALETTE['ink']};
        --sanad-primary: {PALETTE['primary']};
        --sanad-primary-soft: {PALETTE['primary_soft']};
        --sanad-accent: {PALETTE['accent']};
        --sanad-accent-soft: {PALETTE['accent_soft']};
        --sanad-canvas: {PALETTE['canvas']};
        --sanad-surface: {PALETTE['surface']};
        --sanad-border: {PALETTE['border']};
        --sanad-muted: {PALETTE['muted']};
        --sanad-danger: {PALETTE['danger']};
        --sanad-success: {PALETTE['success']};
        --sanad-radius: 12px;
    }}

    /* --- shell ---------------------------------------------------------- */
    /* `direction` alone flips inline flow and the default text alignment, so
       there is no blanket text-align rule here. Forcing alignment on every
       element would also override anything deliberately centred. */
    .stApp {{
        direction: {direction};
        background: var(--sanad-canvas);
        font-family: {_FONT_STACK};
        color: var(--sanad-ink);
    }}
    [data-testid="stSidebar"] {{
        direction: {direction};
        background: var(--sanad-surface);
        border-inline-end: 1px solid var(--sanad-border);
    }}
    [data-testid="stMainBlockContainer"] {{
        padding-top: 2.2rem;
        max-width: 1120px;
    }}
    h1, h2, h3, h4 {{
        color: var(--sanad-primary);
        letter-spacing: -0.01em;
    }}

    /* --- brand ---------------------------------------------------------- */
    .sanad-brand {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.9rem 1rem;
        background: var(--sanad-surface);
        border: 1px solid var(--sanad-border);
        border-radius: var(--sanad-radius);
        margin-bottom: 1.1rem;
    }}
    .sanad-brand img {{ width: 42px; height: 42px; flex: 0 0 auto; }}
    .sanad-brand-text {{ display: flex; flex-direction: column; gap: 0.1rem; min-width: 0; }}
    .sanad-brand-name {{
        font-size: 1.12rem;
        font-weight: 700;
        color: var(--sanad-primary);
        line-height: 1.25;
    }}
    .sanad-brand-sub {{ font-size: 0.8rem; color: var(--sanad-muted); line-height: 1.35; }}
    .sanad-hero {{
        border-inline-start: 4px solid var(--sanad-accent);
        padding-inline-start: 1rem;
        margin-bottom: 1.4rem;
    }}
    .sanad-hero h1 {{ margin: 0 0 0.25rem; font-size: 1.72rem; }}
    .sanad-hero p {{ margin: 0; color: var(--sanad-muted); font-size: 0.95rem; }}

    /* --- panels and states ---------------------------------------------- */
    .sanad-panel {{
        background: var(--sanad-surface);
        border: 1px solid var(--sanad-border);
        border-radius: var(--sanad-radius);
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
    }}
    .sanad-empty {{
        background: var(--sanad-surface);
        border: 1px dashed var(--sanad-border);
        border-radius: var(--sanad-radius);
        padding: 2.4rem 1.5rem;
        text-align: center;
        color: var(--sanad-muted);
    }}
    .sanad-empty h4 {{ margin: 0 0 0.4rem; color: var(--sanad-primary); text-align: center; }}
    .sanad-empty p {{ margin: 0 auto; max-width: 42ch; text-align: center; font-size: 0.92rem; }}
    .sanad-doc-row {{
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        padding: 0.55rem 0.7rem;
        border: 1px solid var(--sanad-border);
        border-radius: 9px;
        background: var(--sanad-canvas);
        margin-bottom: 0.45rem;
    }}
    .sanad-doc-name {{
        font-weight: 600;
        font-size: 0.88rem;
        color: var(--sanad-ink);
        word-break: break-word;
    }}
    .sanad-doc-meta {{ font-size: 0.76rem; color: var(--sanad-muted); }}

    /* --- citations ------------------------------------------------------- */
    .sanad-sources {{
        margin-top: 0.85rem;
        padding: 0.75rem 0.9rem;
        background: #FAFBFA;
        border: 1px solid var(--sanad-border);
        border-inline-start: 3px solid var(--sanad-accent);
        border-radius: 9px;
    }}
    .sanad-sources-title {{
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--sanad-primary);
        margin-bottom: 0.4rem;
    }}
    .sanad-source {{
        display: flex;
        gap: 0.5rem;
        align-items: baseline;
        font-size: 0.85rem;
        padding: 0.16rem 0;
        color: var(--sanad-ink);
    }}
    .sanad-source-marker {{
        flex: 0 0 auto;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.74rem;
        font-weight: 700;
        color: var(--sanad-accent);
        background: var(--sanad-accent-soft);
        border-radius: 5px;
        padding: 0.05rem 0.35rem;
    }}
    .sanad-source-where {{ color: var(--sanad-muted); }}

    /* The verbatim passage behind a citation, rendered by st.text. */
    [data-testid="stExpander"] [data-testid="stText"] {{
        font-size: 0.86rem;
        line-height: 1.65;
        color: var(--sanad-muted);
        background: var(--sanad-canvas);
        border: 1px solid var(--sanad-border);
        border-radius: 8px;
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.7rem;
    }}

    /* --- chat ------------------------------------------------------------ */
    [data-testid="stChatMessage"] {{
        direction: {direction};
        background: var(--sanad-surface);
        border: 1px solid var(--sanad-border);
        border-radius: var(--sanad-radius);
    }}
    [data-testid="stChatInput"] {{ background: var(--sanad-surface); }}

    /* --- controls -------------------------------------------------------- */
    .stButton > button {{
        border-radius: 9px;
        border: 1px solid var(--sanad-primary);
        color: var(--sanad-primary);
        background: var(--sanad-surface);
        font-weight: 600;
    }}
    .stButton > button:hover {{
        border-color: var(--sanad-primary-soft);
        color: var(--sanad-primary-soft);
    }}
    .stButton > button[kind="primary"] {{
        background: var(--sanad-primary);
        color: var(--sanad-surface);
    }}
    [data-testid="stTabs"] button[role="tab"] {{ font-weight: 600; }}

    /* --- footer ---------------------------------------------------------- */
    .sanad-disclaimer {{
        background: #FBF8F2;
        border: 1px solid var(--sanad-accent-soft);
        border-inline-start: 4px solid var(--sanad-accent);
        border-radius: var(--sanad-radius);
        padding: 0.9rem 1.1rem;
        font-size: 0.87rem;
        line-height: 1.6;
        color: var(--sanad-ink);
    }}
    .sanad-disclaimer strong {{ color: var(--sanad-primary); }}

    /* --- responsive ------------------------------------------------------ */
    @media (max-width: 760px) {{
        [data-testid="stMainBlockContainer"] {{ padding-top: 1.2rem; }}
        .sanad-hero h1 {{ font-size: 1.35rem; }}
        .sanad-brand {{ padding: 0.7rem; gap: 0.6rem; }}
        .sanad-brand img {{ width: 34px; height: 34px; }}
        .sanad-brand-name {{ font-size: 1rem; }}
        .sanad-panel {{ padding: 0.85rem 0.9rem; }}
        .sanad-empty {{ padding: 1.6rem 1rem; }}
    }}
    </style>
    """


def brand_block(language: str, *, compact: bool = False) -> str:
    """Markup for the logo lock-up used in the sidebar and the page header."""
    product = BRAND["product_ar"] if is_rtl(language) else BRAND["product_en"]
    tagline = BRAND["tagline_ar"] if is_rtl(language) else BRAND["tagline_en"]
    office = BRAND["office_ar"] if is_rtl(language) else BRAND["office_en"]
    secondary = office if compact else f"{tagline} · {office}"

    return f"""
    <div class="sanad-brand">
        <img src="{logo_data_uri()}" alt="{product}" />
        <div class="sanad-brand-text">
            <div class="sanad-brand-name">{product}</div>
            <div class="sanad-brand-sub">{secondary}</div>
        </div>
    </div>
    """
