from langchain_pollinations._sse import iter_sse_events_from_text


def test_iter_sse_events() -> None:
    text = "data: {\"a\":1}\n\n\ndata: [DONE]\n\n"
    events = list(iter_sse_events_from_text(text))
    assert events[0].data == "{\"a\":1}"
    assert events[1].data == "[DONE]"
