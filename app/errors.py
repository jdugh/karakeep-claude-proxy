"""OpenAI-style error payloads and the exceptions that map to them."""
from __future__ import annotations

from fastapi import HTTPException


class ProxyError(HTTPException):
    """Base class for errors that must be rendered as an OpenAI-style error body."""

    def __init__(self, status_code: int, message: str, error_type: str, code: str, param: str | None = None):
        super().__init__(
            status_code=status_code,
            detail={
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": param,
                    "code": code,
                }
            },
        )


class InvalidRequestError(ProxyError):
    def __init__(self, message: str, param: str | None = None, code: str = "invalid_request"):
        super().__init__(400, message, "invalid_request_error", code, param)


class AuthError(ProxyError):
    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(401, message, "invalid_request_error", "invalid_api_key")


class NotFoundError(ProxyError):
    def __init__(self, message: str, code: str = "not_found"):
        super().__init__(404, message, "invalid_request_error", code)


class PayloadTooLargeError(ProxyError):
    def __init__(self, message: str = "Request payload too large"):
        super().__init__(413, message, "invalid_request_error", "payload_too_large")


class UnsupportedInputError(ProxyError):
    def __init__(self, message: str, code: str = "unsupported_multimodal_input"):
        super().__init__(400, message, "invalid_request_error", code)


class UpstreamTimeoutError(ProxyError):
    def __init__(self, message: str = "Claude Code did not respond in time"):
        super().__init__(504, message, "upstream_error", "claude_timeout")


class RateLimitError(ProxyError):
    def __init__(self, message: str = "Claude subscription rate limit or quota reached"):
        super().__init__(429, message, "upstream_error", "claude_rate_limited")


class UpstreamError(ProxyError):
    def __init__(self, message: str = "Claude inference failed", code: str = "claude_error"):
        super().__init__(502, message, "upstream_error", code)


class ServiceNotReadyError(ProxyError):
    def __init__(self, message: str = "Service is not ready"):
        super().__init__(503, message, "upstream_error", "service_not_ready")
