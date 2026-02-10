from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


def test_with_listeners_contract(monkeypatch: pytest.MonkeyPatch, chat):
    @dataclass
    class FakeResponse:
        payload: dict[str, Any]

        def json(self) -> dict[str, Any]:
            return self.payload

    def post_json_ok(path: str, payload: dict[str, Any], stream: bool = False):
        return FakeResponse(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    monkeypatch.setattr(chat._http, "post_json", post_json_ok)

    # Guardamos eventos sin asumir cuántos start/end ocurren (puede haber runs anidados).
    events: list[str] = []

    # En LangChain v1.2.8, la aridad exacta puede variar según el tracer/run_type.
    # Aceptamos *args/**kwargs para evitar fragilidad.
    def on_start(run, *args, **kwargs):
        events.append("start")

    def on_end(run, *args, **kwargs):
        events.append("end")

    def on_error(run, error, *args, **kwargs):
        events.append("error")

    runnable = chat.with_listeners(on_start=on_start, on_end=on_end, on_error=on_error)
    msg = runnable.invoke("Di: ok")

    assert msg.content.strip().lower() == "ok"

    # Contrato mínimo estable:
    # - Debe haber al menos un start y al menos un end.
    # - No debe haber error.
    assert "start" in events
    assert "end" in events
    assert "error" not in events

    # Invariante de orden (tolerante a runs anidados):
    # el primer "end" no puede ocurrir antes del primer "start".
    first_start = events.index("start")
    first_end = events.index("end")
    assert first_end > first_start

    # Y también debe existir algún "end" posterior a algún "start".
    last_end = len(events) - 1 - list(reversed(events)).index("end")
    assert last_end > first_start

