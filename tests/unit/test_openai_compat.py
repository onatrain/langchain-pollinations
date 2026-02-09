from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_pollinations._openai_compat import lc_messages_to_openai


def test_lc_messages_to_openai_basic() -> None:
    msgs = [
        SystemMessage(content="s"),
        HumanMessage(content="u"),
        ToolMessage(content="tool out", tool_call_id="abc"),
    ]
    out = lc_messages_to_openai(msgs)
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    assert out[2]["role"] == "tool"
    assert out[2]["tool_call_id"] == "abc"

