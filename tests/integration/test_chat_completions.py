import os

import pytest
from langchain_core.messages import HumanMessage

from langchain_pollinations.chat import ChatPollinations

####################################################################################
if os.getenv("POLLINATIONS_HTTP_DEBUG", "").lower() in {"1", "true", "yes", "on"}:
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
    )

    # httpcore es el motor interno de httpx
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.DEBUG)
####################################################################################


@pytest.mark.integration
def test_chat_completions_text() -> None:
    api_key = os.environ["POLLINATIONS_API_KEY"]
    model = ChatPollinations(
        api_key=api_key,
        model="openai",
        temperature=0.2,
        max_tokens=151  # Este modelo impone este número como mínimo de tokens a generar
    )
    res = model.invoke([HumanMessage(content="Responde solo con la palabra: OK")])
    assert "OK" in (res.content or "")

