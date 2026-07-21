"""Exception types for the Adaptyv Foundry client and campaign runner."""

from __future__ import annotations


class AdaptyvError(Exception):
    """Base class for every error raised by this package."""


class AuthError(AdaptyvError):
    """Token missing, malformed, revoked, or lacking the required capability."""


class APIError(AdaptyvError):
    """Foundry returned a non-2xx response that retrying will not fix."""

    def __init__(self, status: int, message: str, body: object = None) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.body = body


class RateLimited(AdaptyvError):
    """Retries were exhausted while the API was still rate limiting us."""


class GuardrailViolation(AdaptyvError):
    """A spend or safety guardrail refused the action.

    Raised *before* any state-changing call reaches Foundry, so seeing this
    means no experiment was created and no invoice exists.
    """
