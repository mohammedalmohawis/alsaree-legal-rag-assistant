"""The Gemini client: authentication, batching, retries, parsing, redaction."""

from __future__ import annotations

import pytest

from sanad.config import GeminiSettings
from sanad.errors import (
    ConfigurationError,
    ModelAuthError,
    ModelError,
    ModelRateLimitError,
    ModelUnreachableError,
)
from sanad.llm.gemini import TASK_DOCUMENT, TASK_QUERY, GeminiClient, normalise
from sanad.llm.transport import NetworkError, TransportResponse
from tests.conftest import FakeTransport, failure, ok, text_response


def client(settings: GeminiSettings, transport, sleeper=lambda _: None) -> GeminiClient:
    return GeminiClient(settings, transport=transport, sleeper=sleeper)


class TestConstruction:
    def test_requires_an_api_key(self) -> None:
        with pytest.raises(ConfigurationError):
            GeminiClient(GeminiSettings(api_key=None), transport=FakeTransport())


class TestAuthentication:
    def test_sends_the_key_as_a_header(self, gemini_settings, transport) -> None:
        client(gemini_settings, transport).generate("hello")

        request = transport.requests[0]
        assert request["headers"]["x-goog-api-key"] == "test-key-do-not-use"

    def test_never_puts_the_key_in_the_url(self, gemini_settings, transport) -> None:
        client(gemini_settings, transport).generate("hello")

        # A key in a query string leaks into proxy and server access logs.
        assert "test-key-do-not-use" not in transport.requests[0]["url"]
        assert "key=" not in transport.requests[0]["url"]

    def test_redacts_the_key_from_error_messages(self, gemini_settings) -> None:
        leaky = FakeTransport([failure(400, "bad key test-key-do-not-use rejected")])

        with pytest.raises(ModelError) as caught:
            client(gemini_settings, leaky).generate("hello")

        assert "test-key-do-not-use" not in str(caught.value)
        assert "***" in str(caught.value)


class TestRetries:
    def test_retries_a_rate_limit_then_succeeds(self, gemini_settings) -> None:
        transport = FakeTransport([failure(429, "slow down"), text_response("done")])

        assert client(gemini_settings, transport).generate("hi") == "done"
        assert transport.call_count == 2

    def test_retries_a_server_error(self, gemini_settings) -> None:
        transport = FakeTransport([failure(503), text_response("recovered")])

        assert client(gemini_settings, transport).generate("hi") == "recovered"

    def test_does_not_retry_a_client_error(self, gemini_settings) -> None:
        transport = FakeTransport([failure(400, "malformed request")])

        with pytest.raises(ModelError, match="malformed request"):
            client(gemini_settings, transport).generate("hi")

        # A 400 will fail identically every time; retrying only wastes the wait.
        assert transport.call_count == 1

    def test_gives_up_after_max_retries(self, gemini_settings) -> None:
        transport = FakeTransport([failure(500), failure(500), failure(500), failure(500)])

        with pytest.raises(ModelError):
            client(gemini_settings, transport).generate("hi")

        assert transport.call_count == gemini_settings.max_retries

    def test_retries_a_network_failure(self, gemini_settings) -> None:
        transport = FakeTransport([NetworkError("connection reset"), text_response("ok")])

        assert client(gemini_settings, transport).generate("hi") == "ok"

    def test_backs_off_exponentially(self, gemini_settings) -> None:
        settings = GeminiSettings(api_key="k", max_retries=3, backoff_seconds=1.0)
        transport = FakeTransport([failure(429), failure(429), text_response("ok")])
        delays: list = []

        GeminiClient(settings, transport=transport, sleeper=delays.append).generate("hi")

        assert delays == [1.0, 2.0]


class TestFailureClassification:
    """A spent retry budget must say what to fix, not dump transport internals."""

    def test_an_unreachable_endpoint_is_reported_as_unreachable(self, gemini_settings) -> None:
        transport = FakeTransport([NetworkError("Connection refused")] * 3)

        with pytest.raises(ModelUnreachableError) as caught:
            client(gemini_settings, transport).embed_documents(["a clause"])

        # The message key drives a "check your connection" sentence in the UI.
        assert caught.value.message_key == "error_model_unreachable"
        assert "Connection refused" in caught.value.technical

    def test_a_rejected_key_is_reported_as_an_auth_failure(self, gemini_settings) -> None:
        for status in (401, 403):
            transport = FakeTransport([failure(status, "caller does not have permission")])

            with pytest.raises(ModelAuthError) as caught:
                client(gemini_settings, transport).generate("hi")

            assert caught.value.message_key == "error_model_auth"

    def test_an_api_key_complaint_on_400_is_also_an_auth_failure(self, gemini_settings) -> None:
        transport = FakeTransport([failure(400, "API key not valid. Please pass a valid API key.")])

        with pytest.raises(ModelAuthError):
            client(gemini_settings, transport).generate("hi")

    def test_exhausted_rate_limiting_is_reported_as_a_rate_limit(self, gemini_settings) -> None:
        transport = FakeTransport([failure(429, "quota exceeded")] * 3)

        with pytest.raises(ModelRateLimitError) as caught:
            client(gemini_settings, transport).generate("hi")

        assert caught.value.message_key == "error_model_rate_limit"

    def test_anything_else_stays_a_plain_model_error(self, gemini_settings) -> None:
        transport = FakeTransport([failure(400, "malformed request")])

        with pytest.raises(ModelError) as caught:
            client(gemini_settings, transport).generate("hi")

        assert type(caught.value) is ModelError
        assert "malformed request" in str(caught.value)

    def test_a_recovered_network_blip_is_not_classified_as_unreachable(
        self, gemini_settings
    ) -> None:
        # Network failure first, then a 400: the final status decides, so a
        # transient blip must not mask the real reason.
        transport = FakeTransport([NetworkError("reset"), failure(400, "malformed request")])

        with pytest.raises(ModelError) as caught:
            client(gemini_settings, transport).generate("hi")

        assert not isinstance(caught.value, ModelUnreachableError)

    def test_the_technical_detail_is_redacted(self, gemini_settings) -> None:
        transport = FakeTransport([NetworkError("failed with key test-key-do-not-use")] * 3)

        with pytest.raises(ModelUnreachableError) as caught:
            client(gemini_settings, transport).generate("hi")

        assert "test-key-do-not-use" not in caught.value.technical
        assert "***" in caught.value.technical

    def test_every_classified_error_is_still_a_model_error(self, gemini_settings) -> None:
        # Callers catch SanadError/ModelError; the subclasses must not escape it.
        transport = FakeTransport([NetworkError("down")] * 3)

        with pytest.raises(ModelError):
            client(gemini_settings, transport).generate("hi")


class TestGeneration:
    def test_joins_multi_part_responses(self, gemini_settings) -> None:
        transport = FakeTransport(
            [ok({"candidates": [{"content": {"parts": [{"text": "one "}, {"text": "two"}]}}]})]
        )

        assert client(gemini_settings, transport).generate("hi") == "one two"

    def test_reports_a_safety_block(self, gemini_settings) -> None:
        transport = FakeTransport([ok({"promptFeedback": {"blockReason": "SAFETY"}})])

        with pytest.raises(ModelError, match="safety filter"):
            client(gemini_settings, transport).generate("hi")

    def test_reports_a_truncated_response(self, gemini_settings) -> None:
        transport = FakeTransport([ok({"candidates": [{"finishReason": "MAX_TOKENS", "content": {}}]})])

        with pytest.raises(ModelError, match="output limit"):
            client(gemini_settings, transport).generate("hi")

    def test_reports_no_candidates(self, gemini_settings) -> None:
        transport = FakeTransport([ok({"candidates": []})])

        with pytest.raises(ModelError, match="no response candidates"):
            client(gemini_settings, transport).generate("hi")

    def test_rejects_a_non_json_body(self, gemini_settings) -> None:
        transport = FakeTransport([TransportResponse(200, None, "<html>gateway</html>")])

        with pytest.raises(ModelError, match="non-JSON"):
            client(gemini_settings, transport).generate("hi")

    def test_passes_a_system_instruction(self, gemini_settings, transport) -> None:
        client(gemini_settings, transport).generate("hi", system_instruction="Be grounded.")

        payload = transport.requests[0]["json"]
        assert payload["systemInstruction"]["parts"][0]["text"] == "Be grounded."

    def test_structured_output_sets_the_schema(self, gemini_settings, transport) -> None:
        schema = {"type": "OBJECT", "properties": {"parties": {"type": "ARRAY"}}}
        client(gemini_settings, transport).generate("hi", response_schema=schema)

        config = transport.requests[0]["json"]["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseSchema"] == schema

    def test_uses_the_configured_temperature(self, gemini_settings, transport) -> None:
        client(gemini_settings, transport).generate("hi")

        assert transport.requests[0]["json"]["generationConfig"]["temperature"] == 0.1


class TestEmbeddings:
    def test_batches_requests(self, gemini_settings, transport) -> None:
        texts = [f"clause number {index}" for index in range(9)]

        vectors = client(gemini_settings, transport).embed_documents(texts)

        assert len(vectors) == 9
        # batch size is 4 in the fixture: 4 + 4 + 1 => three requests, not nine.
        assert len(transport.calls_to(":batchEmbedContents")) == 3

    def test_uses_the_document_task_type(self, gemini_settings, transport) -> None:
        client(gemini_settings, transport).embed_documents(["a clause"])

        request = transport.calls_to(":batchEmbedContents")[0]["json"]["requests"][0]
        assert request["taskType"] == TASK_DOCUMENT
        assert request["outputDimensionality"] == 1536

    def test_uses_the_query_task_type(self, gemini_settings, transport) -> None:
        client(gemini_settings, transport).embed_query("what is the notice period?")

        request = transport.calls_to(":embedContent")[0]["json"]
        assert request["taskType"] == TASK_QUERY

    def test_returns_unit_length_vectors(self, gemini_settings) -> None:
        transport = FakeTransport([ok({"embeddings": [{"values": [3.0, 4.0]}]})])

        vector = client(gemini_settings, transport).embed_documents(["x"])[0]

        assert vector == pytest.approx([0.6, 0.8])

    def test_no_texts_makes_no_request(self, gemini_settings, transport) -> None:
        assert client(gemini_settings, transport).embed_documents([]) == []
        assert transport.call_count == 0

    def test_rejects_a_count_mismatch(self, gemini_settings) -> None:
        # Two texts in, one embedding back: silently accepting this would pair
        # every later chunk with the wrong vector.
        transport = FakeTransport([ok({"embeddings": [{"values": [1.0, 0.0]}]})])

        with pytest.raises(ModelError, match="1 embeddings for 2 chunks"):
            client(gemini_settings, transport).embed_documents(["a", "b"])

    def test_rejects_an_empty_embedding(self, gemini_settings) -> None:
        transport = FakeTransport([ok({"embeddings": [{"values": []}]})])

        with pytest.raises(ModelError, match="empty embedding"):
            client(gemini_settings, transport).embed_documents(["a"])

    def test_rejects_an_empty_query_embedding(self, gemini_settings) -> None:
        transport = FakeTransport([ok({"embedding": {}})])

        with pytest.raises(ModelError, match="empty embedding for the query"):
            client(gemini_settings, transport).embed_query("a")


class TestNormalise:
    def test_scales_to_unit_length(self) -> None:
        assert normalise([3.0, 4.0]) == pytest.approx([0.6, 0.8])

    def test_leaves_a_zero_vector_alone(self) -> None:
        assert normalise([0.0, 0.0]) == [0.0, 0.0]
