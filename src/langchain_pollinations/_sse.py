from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SSEEvent:
    data: str


def iter_sse_events_from_text(text: str) -> Iterator[SSEEvent]:
    """
    Parser mínimo: separa eventos por doble salto de línea y extrae líneas 'data:'.
    Útil para tests unitarios (no para streaming real).
    """
    for block in text.split("\n\n"):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        data_lines = [ln[5:].lstrip() for ln in lines if ln.startswith("data:")]
        if not data_lines:
            continue
        yield SSEEvent(data="\n".join(data_lines))

