"""
This module defines the custom exception classes used throughout the library.
It provides structured error handling for Pollinations API responses and general runtime errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class PollinationsError(RuntimeError):
    """
    Base error class for the Pollinations library.
    All other custom exceptions in this package inherit from this class.
    """


@dataclass(slots=True)
class PollinationsAPIError(PollinationsError):
    """
    Pollinations API Error with structured response support.

    Reflects backend JSON errors with format:
    {
        "status": 400/401/403/500,
        "success": false,
        "error": {
            "code": "BAD_REQUEST" | "UNAUTHORIZED" | "FORBIDDEN" | "INTERNAL_ERROR",
            "message": "...",
            "timestamp": "2026-02-16T...",
            "details": {...},
            "requestId": "req_...",
            "cause": "..."
        }
    }

    Structured fields are automatically parsed from JSON error envelopes for easy debugging.
    """

    status_code: int
    message: str
    body: str | None = None

    # Campos estructurados del API (opcionales para backward compatibility)
    error_code: str | None = None
    request_id: str | None = None
    timestamp: str | None = None
    details: dict[str, Any] | None = None
    cause: Any | None = None

    def __str__(self) -> str:
        """
        Return a string representation of the API error including status code and details.

        Returns:
            A formatted string containing status code, error code, and message.
        """
        parts = [f"PollinationsAPIError(status_code={self.status_code}"]
        if self.error_code:
            parts.append(f", code={self.error_code!r}")
        parts.append(f", message={self.message!r}")
        if self.request_id:
            parts.append(f", request_id={self.request_id!r}")
        if self.body:
            parts.append(f", body={len(self.body)} chars")
        parts.append(")")
        return "".join(parts)

    def __repr__(self) -> str:
        """
        Return a developer-friendly representation of the error object.

        Returns:
            A string showing all internal fields of the exception.
        """
        return (
            f"PollinationsAPIError("
            f"status_code={self.status_code}, "
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"request_id={self.request_id!r}, "
            f"timestamp={self.timestamp!r}, "
            f"details={self.details!r}, "
            f"cause={self.cause!r}, "
            f"body={'...' if self.body else None})"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the error instance into a dictionary for structured logging.

        Returns:
            A dictionary containing all error attributes and metadata.
        """
        return {
            "status_code": self.status_code,
            "message": self.message,
            "error_code": self.error_code,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "details": self.details,
            "cause": self.cause,
            "body": self.body,
        }

    @property
    def is_client_error(self) -> bool:
        """
        Check if the error corresponds to a 4xx client-side issue.

        Returns:
            True if the HTTP status code is between 400 and 499.
        """
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """
        Check if the error corresponds to a 5xx server-side issue.

        Returns:
            True if the HTTP status code is between 500 and 599.
        """
        return 500 <= self.status_code < 600

    @property
    def is_auth_error(self) -> bool:
        """
        Determine if the error is related to authentication or authorization.

        Returns:
            True if the status code is 401 (Unauthorized) or 403 (Forbidden).
        """
        return self.status_code in (401, 403)

    @property
    def is_validation_error(self) -> bool:
        """
        Determine if the error is due to validation of parameters sent to the API.

        Returns:
            True if the status is 400 and the error code is 'BAD_REQUEST'.
        """
        return self.status_code == 400 and self.error_code == "BAD_REQUEST"
