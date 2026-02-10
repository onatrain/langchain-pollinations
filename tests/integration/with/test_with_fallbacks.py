from __future__ import annotations

import pytest


@pytest.mark.integration
def test_with_fallbacks_contract(pollinations_api_key: str):
    from langchain_pollinations.chat import ChatPollinations

    primary = ChatPollinations(
        api_key=pollinations_api_key,
        base_url="http://127.0.0.1:1",
        timeout_s=0.2,
        model="openai",
        temperature=0,
    )
    fallback = ChatPollinations(
        api_key=pollinations_api_key,
        model="openai",
        temperature=0,
    )

    runnable = primary.with_fallbacks([fallback])
    msg = runnable.invoke("Responde exactamente con la palabra: ok")

    assert hasattr(msg, "content")
    assert msg.content.strip().lower() == "ok"

