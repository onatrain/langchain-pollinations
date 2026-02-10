from __future__ import annotations

from dataclasses import dataclass


def test_with_config_contract(monkeypatch, chat):
    @dataclass
    class FakeResponse:
        payload: dict

        def json(self):
            return self.payload

    seen = {"payload": None}

    def capture_post_json(path, payload, stream=False):
        seen["payload"] = payload
        return FakeResponse(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    monkeypatch.setattr(chat._http, "post_json", capture_post_json)

    runnable = chat.with_config(tags=["contract"], metadata={"k": "v"})
    msg = runnable.invoke("Di: ok")

    assert msg.content.strip().lower() == "ok"
    assert isinstance(seen["payload"], dict)
    assert "tags" not in seen["payload"]
    assert "metadata" not in seen["payload"]

