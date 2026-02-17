from __future__ import annotations
from dataclasses import dataclass
from typing import Any


class PollinationsError(RuntimeError):
    """Error base de la librería."""


@dataclass(slots=True)
class PollinationsAPIError(PollinationsError):
    """
    Error de API de Pollinations con soporte para respuestas estructuradas.

    Cuando el backend retorna un error JSON con formato:
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

    Los campos estructurados se parsean automáticamente para facilitar debugging.
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
        """Convierte el error a dict para logging estructurado."""
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
        """True si es un error 4xx (problema del cliente)."""
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """True si es un error 5xx (problema del servidor)."""
        return 500 <= self.status_code < 600

    @property
    def is_auth_error(self) -> bool:
        """True si es un error de autenticación (401) o autorización (403)."""
        return self.status_code in (401, 403)

    @property
    def is_validation_error(self) -> bool:
        """True si es un error de validación (400 con BAD_REQUEST)."""
        return self.status_code == 400 and self.error_code == "BAD_REQUEST"
