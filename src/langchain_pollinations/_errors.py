from __future__ import annotations

from dataclasses import dataclass


class PollinationsError(RuntimeError):
    """Error base de la librería."""


@dataclass(slots=True)
class PollinationsAPIError(PollinationsError):
    status_code: int
    message: str
    body: str | None = None

    def __str__(self) -> str:
        extra = f" body={self.body!r}" if self.body else ""
        return f"PollinationsAPIError(status_code={self.status_code}, message={self.message!r}{extra})"
