"""A minimal HTTP seam between Sanad and the Gemini REST API.

The client depends on this small interface rather than on ``requests``
directly, so the whole model layer — batching, retries, response parsing and
error handling — is unit-testable without a network or an API key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:  # Protocol landed in typing in 3.8; the fallback keeps type-checking optional.
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class TransportResponse:
    """A protocol-agnostic HTTP response."""

    status_code: int
    payload: dict[str, Any] | None = None
    text: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def retryable(self) -> bool:
        """Whether retrying this response could plausibly succeed.

        429 is rate limiting and 5xx are server-side faults; both are transient.
        A 4xx such as 400 or 403 means the request or key is wrong, and
        retrying it only wastes the user's time.
        """
        return self.status_code == 429 or 500 <= self.status_code < 600

    def error_message(self) -> str:
        """Best available human-readable reason for a failed response."""
        if isinstance(self.payload, dict):
            error = self.payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
        if self.text.strip():
            return self.text.strip()[:400]
        return f"HTTP {self.status_code}"


class Transport(Protocol):  # pragma: no cover - structural interface only
    """Anything that can POST JSON and return a :class:`TransportResponse`."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: int,
    ) -> TransportResponse:
        ...


class NetworkError(RuntimeError):
    """Raised by a transport when the request never reached the server."""


class RequestsTransport:
    """The production transport, backed by ``requests``."""

    def __init__(self, session: Any | None = None) -> None:
        import requests  # imported lazily so tests need no network stack

        self._requests = requests
        self._session = session or requests.Session()

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: int,
    ) -> TransportResponse:
        try:
            response = self._session.post(
                url, headers=dict(headers), json=dict(json), timeout=timeout
            )
        except self._requests.RequestException as error:
            raise NetworkError(str(error)) from error

        try:
            payload = response.json()
        except ValueError:
            payload = None
        return TransportResponse(
            status_code=response.status_code,
            payload=payload if isinstance(payload, dict) else None,
            text=response.text or "",
        )
