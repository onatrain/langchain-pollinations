"""
Minimal parser for Server-Sent Events (SSE) that extracts data fields from text blocks.
Designed primarily for simple string-based event processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """
    Data structure representing a single Server-Sent Event (SSE).
    Stores the raw string content associated with the 'data' field.
    """

    data: str


def iter_sse_events_from_text(text: str) -> Iterator[SSEEvent]:
    """
    Parse SSE events from a text block by splitting on double newlines.

    Args:
        text: The raw string containing one or multiple SSE events.

    Yields:
        SSEEvent objects containing the reconstructed data fields.
    """
    for block in text.split("\n\n"):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        data_lines = [ln[5:].lstrip() for ln in lines if ln.startswith("data:")]
        if not data_lines:
            continue
        yield SSEEvent(data="\n".join(data_lines))
