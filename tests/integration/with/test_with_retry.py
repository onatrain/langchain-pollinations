from __future__ import annotations

from dataclasses import dataclass


def test_with_retry_contract(monkeypatch, chat):
    from langchain_pollinations._errors import PollinationsAPIError

    @dataclass
    class FakeResponse:
        payload: dict

        def json(self):
            return self.payload

    calls = {"n": 0}

    def flaky_post_json(path, payload, stream=False):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PollinationsAPIError(status_code=503, message="Service Unavailable", body="upstream")
        return FakeResponse(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "pong"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    monkeypatch.setattr(chat._http, "post_json", flaky_post_json)

    runnable = chat.with_retry(stop_after_attempt=3)
    msg = runnable.invoke("Responde solo con: pong")

    assert calls["n"] == 3
    assert hasattr(msg, "content")
    assert msg.content.strip().lower() == "pong"

