"""Gemini REST client: generation and batched embeddings.

Three things distinguish this from the single-call helper it replaces:

*Batching.* Embeddings go through ``:batchEmbedContents``, so indexing a
100-chunk contract costs two requests instead of a hundred.

*Task-typed embeddings.* Documents are embedded with ``RETRIEVAL_DOCUMENT`` and
queries with ``RETRIEVAL_QUERY``. Gemini produces asymmetric vectors for the two
roles, and using the right one is a free, material retrieval-quality gain.

*Retries and redaction.* Rate limits and 5xx responses are retried with
exponential backoff; every error message is scrubbed of the API key before it
can reach a log or the screen.

Authentication uses the ``x-goog-api-key`` header — never a query parameter,
which would leak the key into proxy and server logs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from typing import Any, Callable

from sanad.config import GeminiSettings
from sanad.errors import (
    ConfigurationError,
    ModelAuthError,
    ModelError,
    ModelRateLimitError,
    ModelUnreachableError,
)
from sanad.llm.transport import NetworkError, RequestsTransport, Transport, TransportResponse

#: Task types documented for gemini-embedding models.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

Vector = list[float]


def normalise(vector: Sequence[float]) -> Vector:
    """Scale a vector to unit length so a dot product is a cosine similarity.

    Required, not cosmetic: requesting a reduced ``outputDimensionality``
    truncates the tail of the embedding, which leaves the remaining vector
    slightly off unit norm. Re-normalising restores comparability.
    """
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return [0.0] * len(vector)
    return [value / magnitude for value in vector]


def _batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    step = max(1, size)
    for start in range(0, len(items), step):
        yield items[start : start + step]


class GeminiClient:
    """A thin, typed wrapper over the two Gemini endpoints Sanad uses."""

    def __init__(
        self,
        settings: GeminiSettings,
        transport: Transport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.has_api_key:
            raise ConfigurationError("No Gemini API key configured.")
        self._settings = settings
        self._transport = transport or RequestsTransport()
        self._sleep = sleeper

    # --- transport ------------------------------------------------------------

    def _redact(self, message: str) -> str:
        """Remove the API key from a message before it is shown or logged."""
        key = self._settings.api_key
        if key and key in message:
            message = message.replace(key, "***")
        return message

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to ``path``, retrying transient failures with exponential backoff."""
        url = f"{self._settings.api_base.rstrip('/')}/{path.lstrip('/')}"
        headers = {
            "x-goog-api-key": self._settings.api_key or "",
            "Content-Type": "application/json",
        }
        attempts = max(1, self._settings.max_retries)
        last_error = "unknown error"
        last_status: int | None = None
        unreachable = False

        for attempt in range(attempts):
            try:
                response: TransportResponse = self._transport.post(
                    url, headers=headers, json=payload, timeout=self._settings.timeout_seconds
                )
            except NetworkError as error:
                last_error = f"network error: {error}"
                unreachable = True
                if attempt == attempts - 1:
                    break
                self._sleep(self._settings.backoff_seconds * (2**attempt))
                continue

            unreachable = False
            if response.ok:
                if response.payload is None:
                    raise ModelError(self._redact("Gemini returned a non-JSON response."))
                return response.payload

            last_error = response.error_message()
            last_status = response.status_code
            if not response.retryable or attempt == attempts - 1:
                break
            self._sleep(self._settings.backoff_seconds * (2**attempt))

        raise self._failure(unreachable, last_status, last_error)

    def _failure(
        self, unreachable: bool, status: int | None, detail: str
    ) -> ModelError:
        """Turn a spent retry budget into the most actionable error available.

        The caller sees a message describing what to fix; the raw transport text
        travels along as ``technical`` rather than being the message itself.
        """
        technical = self._redact(detail)

        if unreachable:
            return ModelUnreachableError(technical, technical=technical)
        if status in (401, 403) or (status == 400 and "api key" in detail.lower()):
            return ModelAuthError(technical, technical=technical)
        if status == 429:
            return ModelRateLimitError(technical, technical=technical)
        return ModelError(f"Gemini request failed: {technical}", technical=technical)

    # --- generation -----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float | None = None,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """Generate text, optionally constrained to a JSON schema.

        Passing ``response_schema`` switches the model into structured-output
        mode, which is how key-fact extraction gets a parseable result instead
        of prose that has to be scraped.
        """
        generation_config: dict[str, Any] = {
            "temperature": self._settings.temperature if temperature is None else temperature
        }
        if max_output_tokens:
            generation_config["maxOutputTokens"] = max_output_tokens
        if response_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = response_schema

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        response = self._post(f"models/{self._settings.chat_model}:generateContent", payload)
        return self._extract_text(response)

    def _extract_text(self, response: dict[str, Any]) -> str:
        """Pull the generated text out of a ``generateContent`` response.

        Blocked and truncated responses are reported as such rather than as an
        empty string, so the UI can tell the user what actually happened.
        """
        feedback = response.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            raise ModelError(f"The request was blocked by a safety filter ({feedback['blockReason']}).")

        candidates = response.get("candidates") or []
        if not candidates:
            raise ModelError("Gemini returned no response candidates.")

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        ).strip()

        if not text:
            reason = candidate.get("finishReason") or "unknown"
            if reason == "MAX_TOKENS":
                raise ModelError("The response exceeded the output limit before any text was produced.")
            raise ModelError(f"Gemini returned an empty response (finishReason={reason}).")
        return text

    # --- embeddings -----------------------------------------------------------

    def _embed_request(self, text: str, task_type: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": f"models/{self._settings.embedding_model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }
        if self._settings.embedding_dimensions:
            request["outputDimensionality"] = self._settings.embedding_dimensions
        return request

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed many chunks using batched requests.

        Raises :class:`ModelError` if the API returns a different number of
        embeddings than texts sent, because a silent misalignment would attach
        every chunk to the wrong vector and corrupt every later citation.
        """
        if not texts:
            return []

        vectors: list[Vector] = []
        for batch in _batched(list(texts), self._settings.embedding_batch_size):
            payload = {"requests": [self._embed_request(text, TASK_DOCUMENT) for text in batch]}
            response = self._post(
                f"models/{self._settings.embedding_model}:batchEmbedContents", payload
            )
            embeddings = response.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise ModelError(
                    "Gemini returned "
                    f"{len(embeddings) if isinstance(embeddings, list) else 0} embeddings "
                    f"for {len(batch)} chunks."
                )
            for item in embeddings:
                values = (item or {}).get("values")
                if not values:
                    raise ModelError("Gemini returned an empty embedding for a chunk.")
                vectors.append(normalise(values))
        return vectors

    def embed_query(self, text: str) -> Vector:
        """Embed a single search query with the query-side task type."""
        payload = self._embed_request(text, TASK_QUERY)
        response = self._post(f"models/{self._settings.embedding_model}:embedContent", payload)
        values = (response.get("embedding") or {}).get("values")
        if not values:
            raise ModelError("Gemini returned an empty embedding for the query.")
        return normalise(values)
