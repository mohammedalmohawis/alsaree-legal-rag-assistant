"""Central, immutable configuration for Sanad.

Every tunable lives here as a frozen dataclass with a documented default, and
every default can be overridden by an environment variable. Nothing else in the
package reads ``os.environ`` directly, which keeps the settings surface small
and makes the whole pipeline trivial to configure in tests.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv

T = TypeVar("T")

# --- Brand identity -----------------------------------------------------------
# Sanad (سند) is Arabic for a legal instrument or deed. The platform is operated
# by Abdulaziz Al-Saree Law Office; both names are shown together in the UI.

BRAND = {
    "product_en": "Sanad",
    "product_ar": "سند",
    "tagline_en": "Legal Document Intelligence",
    "tagline_ar": "منصة تحليل المستندات القانونية",
    "office_en": "Abdulaziz Al-Saree Law Office",
    "office_ar": "مكتب عبد العزيز السريع للمحاماة",
}

#: Sanad's own palette: deep emerald with a bronze accent, on a cool paper canvas.
PALETTE = {
    "ink": "#0B1F1C",
    "primary": "#0E3B33",
    "primary_soft": "#14544A",
    "accent": "#B07D46",
    "accent_soft": "#E8D9C2",
    "canvas": "#F4F6F5",
    "surface": "#FFFFFF",
    "border": "#DCE4E1",
    "muted": "#5C6B67",
    "danger": "#8C2F26",
    "success": "#1F6F4A",
}

SUPPORTED_LANGUAGES = ("en", "ar")


def _read(
    env: Mapping[str, str],
    name: str,
    default: T,
    cast: Callable[[str], Any] = str,
) -> T:
    """Return ``env[name]`` cast to the default's type, or the default itself.

    A malformed override falls back to the default rather than crashing the
    app at import time: a bad ``SANAD_TOP_K`` should not stop the office from
    working.
    """
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())  # type: ignore[return-value]
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class GeminiSettings:
    """Model selection, transport behaviour and retry policy."""

    api_key: str | None = None
    chat_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-2"
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_seconds: int = 60
    max_retries: int = 3
    backoff_seconds: float = 1.5
    #: gemini-embedding-2 accepts 128-3072; 1536 keeps recall high at half the
    #: memory of the full 3072-dimension vector.
    embedding_dimensions: int = 1536
    #: Number of chunks per :batchEmbedContents call. The API documents no hard
    #: limit, so this is a conservative value that keeps request bodies small.
    embedding_batch_size: int = 64
    #: Deterministic by default: legal answers should not vary between runs.
    temperature: float = 0.1

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class ChunkingSettings:
    """Chunk geometry, in characters.

    Legal retrieval is clause-level, so targets are an order of magnitude
    smaller than a naive full-page split. ``target_chars`` is the size the
    packer aims for; ``max_chars`` is the hard ceiling that forces a split even
    mid-paragraph.
    """

    target_chars: int = 1200
    max_chars: int = 1800
    min_chars: int = 250
    overlap_chars: int = 180


@dataclass(frozen=True)
class RetrievalSettings:
    """Hybrid retrieval geometry.

    ``candidate_pool`` chunks are pulled from each of the dense and lexical
    indexes, fused, then reduced to ``top_k`` by relevance-diversity selection.
    """

    top_k: int = 6
    candidate_pool: int = 24
    #: Reciprocal-rank-fusion damping. 60 is the value from the original RRF
    #: paper and needs no per-corpus tuning.
    rrf_k: int = 60
    #: 1.0 = pure relevance, 0.0 = pure diversity. 0.72 keeps near-duplicate
    #: boilerplate clauses from crowding out the answer.
    mmr_lambda: float = 0.72
    #: Chunks scoring below this fused floor are dropped rather than padded in.
    min_score: float = 0.0


@dataclass(frozen=True)
class AppSettings:
    """The complete resolved configuration for one process."""

    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    default_language: str = "en"
    max_upload_mb: int = 50


def resolve_api_key(
    env: Mapping[str, str] | None = None,
    secrets: Mapping[str, str] | None = None,
) -> str | None:
    """Find the Gemini key, preferring deployment secrets over the environment.

    ``secrets`` is Streamlit's secrets mapping when running on Community Cloud.
    Accessing it can raise if no secrets file exists, so callers pass an already
    materialised mapping and every lookup here is total.
    """
    env = os.environ if env is None else env
    for source in (secrets or {}, env):
        for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "API_KEY"):
            value = source.get(name)
            if value and value.strip():
                return value.strip()
    return None


def load_settings(
    env: Mapping[str, str] | None = None,
    secrets: Mapping[str, str] | None = None,
    *,
    read_dotenv: bool = True,
) -> AppSettings:
    """Build :class:`AppSettings` from ``.env``, the environment and secrets.

    Passing ``env``/``secrets`` explicitly (as the tests do) makes the whole
    configuration layer hermetic.
    """
    if read_dotenv and env is None:
        load_dotenv()
    env = os.environ if env is None else env

    gemini = GeminiSettings(
        api_key=resolve_api_key(env, secrets),
        chat_model=_read(env, "SANAD_CHAT_MODEL", GeminiSettings.chat_model),
        embedding_model=_read(
            env, "SANAD_EMBEDDING_MODEL", GeminiSettings.embedding_model
        ),
        api_base=_read(env, "SANAD_API_BASE", GeminiSettings.api_base),
        timeout_seconds=_read(
            env, "SANAD_TIMEOUT_SECONDS", GeminiSettings.timeout_seconds, int
        ),
        max_retries=_read(env, "SANAD_MAX_RETRIES", GeminiSettings.max_retries, int),
        embedding_dimensions=_read(
            env,
            "SANAD_EMBEDDING_DIMENSIONS",
            GeminiSettings.embedding_dimensions,
            int,
        ),
        embedding_batch_size=_read(
            env,
            "SANAD_EMBEDDING_BATCH_SIZE",
            GeminiSettings.embedding_batch_size,
            int,
        ),
        temperature=_read(
            env, "SANAD_TEMPERATURE", GeminiSettings.temperature, float
        ),
    )
    chunking = ChunkingSettings(
        target_chars=_read(
            env, "SANAD_CHUNK_TARGET_CHARS", ChunkingSettings.target_chars, int
        ),
        max_chars=_read(env, "SANAD_CHUNK_MAX_CHARS", ChunkingSettings.max_chars, int),
        min_chars=_read(env, "SANAD_CHUNK_MIN_CHARS", ChunkingSettings.min_chars, int),
        overlap_chars=_read(
            env, "SANAD_CHUNK_OVERLAP_CHARS", ChunkingSettings.overlap_chars, int
        ),
    )
    retrieval = RetrievalSettings(
        top_k=_read(env, "SANAD_TOP_K", RetrievalSettings.top_k, int),
        candidate_pool=_read(
            env, "SANAD_CANDIDATE_POOL", RetrievalSettings.candidate_pool, int
        ),
        mmr_lambda=_read(
            env, "SANAD_MMR_LAMBDA", RetrievalSettings.mmr_lambda, float
        ),
    )
    language = _read(env, "SANAD_DEFAULT_LANGUAGE", AppSettings.default_language)
    if language not in SUPPORTED_LANGUAGES:
        language = AppSettings.default_language

    return AppSettings(
        gemini=gemini,
        chunking=chunking,
        retrieval=retrieval,
        default_language=language,
        max_upload_mb=_read(env, "SANAD_MAX_UPLOAD_MB", AppSettings.max_upload_mb, int),
    )
