"""Typed errors so the UI can show an actionable message instead of a traceback.

Every error carries a translation key from :mod:`sanad.i18n` so the same
exception renders correctly in both Arabic and English.
"""

from __future__ import annotations


class SanadError(Exception):
    """Base class for every error this application raises deliberately."""

    #: Translation key used to render the error to the end user.
    message_key = "error_generic"

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail)
        self.detail = detail


class ConfigurationError(SanadError):
    """Raised when required configuration (typically the API key) is absent."""

    message_key = "error_api_key_missing"


class DocumentReadError(SanadError):
    """Raised when a single upload cannot be parsed.

    The filename is kept separate from the underlying reason so the UI can
    report *which* upload failed without losing the technical detail.
    """

    message_key = "error_document_read"

    def __init__(self, filename: str, detail: str = "") -> None:
        super().__init__(detail)
        self.filename = filename


class UnsupportedFormatError(DocumentReadError):
    """Raised for uploads whose extension is outside the supported set."""

    message_key = "error_unsupported_format"


class EmptyDocumentError(DocumentReadError):
    """Raised when a file parses cleanly but contains no extractable text.

    Scanned PDFs without an OCR layer are the common cause, which is why the
    translated message points the user towards OCR.
    """

    message_key = "error_empty_document"


class ModelError(SanadError):
    """Raised when the Gemini API rejects a request or returns nothing usable.

    ``technical`` carries the underlying transport or API text. The subclasses
    below translate to a message that tells the reader what to *do*, while the
    technical string stays attached for the detail panel and for logs.
    """

    message_key = "error_model"

    def __init__(self, detail: str = "", *, technical: str = "") -> None:
        super().__init__(detail)
        self.technical = technical or detail


class ModelUnreachableError(ModelError):
    """Raised when the Gemini endpoint could not be contacted at all.

    A dead network, an offline machine, DNS failure, or a misconfigured
    ``SANAD_API_BASE`` pointing somewhere nothing is listening.
    """

    message_key = "error_model_unreachable"


class ModelAuthError(ModelError):
    """Raised when Gemini rejects the API key."""

    message_key = "error_model_auth"


class ModelRateLimitError(ModelError):
    """Raised when Gemini is still rate limiting after every retry."""

    message_key = "error_model_rate_limit"


class RetrievalError(SanadError):
    """Raised when a query is issued against an empty or unusable index."""

    message_key = "error_retrieval"
