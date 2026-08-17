"""Model access: the Gemini REST client, its HTTP seam, and all prompts."""

from __future__ import annotations

from sanad.llm.gemini import TASK_DOCUMENT, TASK_QUERY, GeminiClient, normalise
from sanad.llm.transport import (
    NetworkError,
    RequestsTransport,
    Transport,
    TransportResponse,
)

__all__ = [
    "GeminiClient",
    "NetworkError",
    "RequestsTransport",
    "TASK_DOCUMENT",
    "TASK_QUERY",
    "Transport",
    "TransportResponse",
    "normalise",
]
