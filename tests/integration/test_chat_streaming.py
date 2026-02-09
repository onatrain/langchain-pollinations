import os

import pytest
from langchain_core.messages import HumanMessage

from langchain_pollinations.chat import ChatPollinations

##################################################################
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
)

# httpcore es el motor interno de httpx
logging.getLogger("httpcore").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
##################################################################


def _chunk_to_text(chunk: object) -> str:
    # LangChain puede devolver distintos tipos según versión:
    # - AIMessageChunk (tiene .content)
    # - ChatGenerationChunk (tiene .message.content)
    # - str (texto directo)
    if isinstance(chunk, str):
        return chunk
    content = getattr(chunk, "content", None)
    if isinstance(content, str) and content:
        return content
    msg = getattr(chunk, "message", None)
    if msg is not None:
        msg_content = getattr(msg, "content", None)
        if isinstance(msg_content, str) and msg_content:
            return msg_content
    return ""


@pytest.mark.integration
def test_chat_streaming_collect() -> None:
    api_key = os.environ["POLLINATIONS_API_KEY"]

    model = ChatPollinations(
        api_key=api_key,
        model="openai",
        temperature=0.2,
    )

    pieces: list[str] = []
    for chunk in model.stream([HumanMessage(content="Di: hola")]):
        text = _chunk_to_text(chunk)
        if text:
            pieces.append(text)

    final = "".join(pieces).strip()
    assert final