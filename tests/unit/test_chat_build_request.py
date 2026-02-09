import pytest

from langchain_core.messages import HumanMessage
from langchain_pollinations.chat import ChatPollinations


def test_chat_build_payload_includes_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLLINATIONS_API_KEY", "x")
    m = ChatPollinations(model="openai", temperature=0.2)
    payload = m._build_payload([HumanMessage(content="hi")], stop=None)  # noqa: SLF001
    assert payload["model"] == "openai"
    assert payload["messages"][0]["role"] == "user"
    assert payload["temperature"] == 0.2

