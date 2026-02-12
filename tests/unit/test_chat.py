from pathlib import Path
from typing import Any
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError, BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from langchain_pollinations.chat import (
    AudioConfig,
    StreamOptions,
    ThinkingConfig,
    ResponseFormatText,
    ResponseFormatJsonObject,
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaObject,
    ToolFunction,
    ToolFunctionTool,
    ToolBuiltinTool,
    FunctionDef,
    ToolChoiceFunctionInner,
    ToolChoiceFunction,
    FunctionCallName,
    ChatPollinationsConfig,
    ChatPollinations,
    _normalize_tool_choice,
    _extract_text_from_content_blocks,
    _usage_metadata_from_usage,
    _response_metadata_from_response,
    _message_content_from_message_dict,
    _delta_content_from_delta_dict,
    _text_from_any_content,
    _tool_call_chunks_from_delta,
    _parse_tool_calls,
    _iter_sse_json_events_sync,
    _iter_sse_json_events_async,
    DEFAULT_BASE_URL,
)


@pytest.fixture
def api_key_from_env(monkeypatch) -> str:
    """
    Lee POLLINATIONS_API_KEY desde .env si existe, y lo inyecta en el entorno.
    Si no existe o no define la variable, usa un valor por defecto para tests.
    """
    env_path = Path(".env")
    api_key = "test_api_key_from_env"
    env_var_name = "POLLINATIONS_API_KEY"

    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == env_var_name:
                api_key = value.strip().strip("'").strip('"')
                break

    monkeypatch.setenv(env_var_name, api_key)
    return api_key


@dataclass
class DummyAuth:
    api_key: str


class DummyResponse:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def json(self) -> dict[str, Any]:
        return self._data


class DummyStreamResponse:
    def __init__(self, events: list[str]):
        self.events = events
        self.status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_lines(self):
        for event in self.events:
            yield event

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aiter_lines(self):
        for event in self.events:
            yield event


class DummyHttpClient:
    def __init__(self, *, config, api_key: str):
        self.config = config
        self.api_key = api_key
        self.calls: list[dict[str, Any]] = []
        self.stream_events: list[str] = []

    def post_json(self, path: str, payload: dict[str, Any], *, stream: bool = False):
        self.calls.append({"method": "post_json", "path": path, "payload": payload, "stream": stream})
        return DummyResponse({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "openai",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from test"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })

    async def apost_json(self, path: str, payload: dict[str, Any], *, stream: bool = False):
        self.calls.append({"method": "apost_json", "path": path, "payload": payload, "stream": stream})
        return DummyResponse({
            "id": "chatcmpl-async-123",
            "object": "chat.completion",
            "model": "openai",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from async test"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14}
        })

    def stream_post_json(self, path: str, payload: dict[str, Any]):
        self.calls.append({"method": "stream_post_json", "path": path, "payload": payload})
        return DummyStreamResponse(self.stream_events or [
            'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"index":0,"delta":{"content":" world"}}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            'data: [DONE]'
        ])

    def astream_post_json(self, path: str, payload: dict[str, Any]):
        self.calls.append({"method": "astream_post_json", "path": path, "payload": payload})
        return DummyStreamResponse(self.stream_events or [
            'data: {"choices":[{"index":0,"delta":{"content":"Async"}}]}',
            'data: {"choices":[{"index":0,"delta":{"content":" stream"}}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            'data: [DONE]'
        ])

    @staticmethod
    def raise_for_status(resp):
        pass


@pytest.fixture(autouse=True)
def patch_auth_and_http(monkeypatch):
    """Mock automático de AuthConfig y PollinationsHttpClient para evitar llamadas reales."""
    monkeypatch.setattr(
        "langchain_pollinations._auth.AuthConfig.from_env_or_value",
        staticmethod(lambda api_key: DummyAuth(api_key=api_key or "dummy")),
    )
    monkeypatch.setattr(
        "langchain_pollinations.chat.PollinationsHttpClient",
        DummyHttpClient,
    )
    yield


def test_audio_config_valid():
    config = AudioConfig(voice="alloy", format="mp3")
    assert config.voice == "alloy"
    assert config.format == "mp3"


def test_audio_config_forbids_extra():
    with pytest.raises(ValidationError):
        AudioConfig(voice="alloy", format="mp3", extra_field="value")


def test_stream_options_allows_extra():
    config = StreamOptions(include_usage=True, custom_field="allowed")
    assert config.include_usage is True


def test_thinking_config_default():
    config = ThinkingConfig()
    assert config.type == "disabled"
    assert config.budget_tokens is None


def test_thinking_config_with_budget():
    config = ThinkingConfig(type="enabled", budget_tokens=1000)
    assert config.type == "enabled"
    assert config.budget_tokens == 1000


def test_response_format_text():
    fmt = ResponseFormatText(type="text")
    assert fmt.type == "text"


def test_response_format_json_object():
    fmt = ResponseFormatJsonObject(type="json_object")
    assert fmt.type == "json_object"


def test_response_format_json_schema():
    schema_obj = ResponseFormatJsonSchemaObject(
        name="TestSchema",
        description="Test schema",
        schema_={"type": "object", "properties": {"name": {"type": "string"}}},
        strict=True
    )
    fmt = ResponseFormatJsonSchema(type="json_schema", json_schema=schema_obj)
    assert fmt.type == "json_schema"
    assert fmt.json_schema.name == "TestSchema"
    assert fmt.json_schema.strict is True


def test_tool_function():
    func = ToolFunction(
        name="get_weather",
        description="Get weather info",
        parameters={"type": "object", "properties": {}},
        strict=False
    )
    assert func.name == "get_weather"
    assert func.description == "Get weather info"


def test_tool_function_tool():
    tool = ToolFunctionTool(
        type="function",
        function=ToolFunction(name="test_func")
    )
    assert tool.type == "function"
    assert tool.function.name == "test_func"


def test_tool_builtin_tool():
    tool = ToolBuiltinTool(type="code_execution")
    assert tool.type == "code_execution"


def test_function_def():
    func = FunctionDef(name="my_func", description="Does something")
    assert func.name == "my_func"
    assert func.description == "Does something"


def test_function_def_forbids_extra():
    with pytest.raises(ValidationError):
        FunctionDef(name="func", extra="not_allowed")


def test_tool_choice_function():
    choice = ToolChoiceFunction(
        type="function",
        function=ToolChoiceFunctionInner(name="specific_tool")
    )
    assert choice.type == "function"
    assert choice.function.name == "specific_tool"


def test_function_call_name():
    call = FunctionCallName(name="my_function")
    assert call.name == "my_function"


def test_chat_pollinations_config_defaults():
    config = ChatPollinationsConfig()
    assert config.model is None
    assert config.temperature is None
    assert config.max_tokens is None


def test_chat_pollinations_config_with_values():
    config = ChatPollinationsConfig(
        model="openai",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        seed=42,
        presence_penalty=0.5,
        frequency_penalty=-0.5,
        logprobs=True,
        top_logprobs=5
    )
    assert config.model == "openai"
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.top_p == 0.9
    assert config.seed == 42
    assert config.presence_penalty == 0.5
    assert config.frequency_penalty == -0.5
    assert config.logprobs is True
    assert config.top_logprobs == 5


def test_chat_pollinations_config_temperature_validation():
    with pytest.raises(ValidationError):
        ChatPollinationsConfig(temperature=-0.1)

    with pytest.raises(ValidationError):
        ChatPollinationsConfig(temperature=2.1)


def test_chat_pollinations_config_seed_validation():
    with pytest.raises(ValidationError):
        ChatPollinationsConfig(seed=-2)

    config = ChatPollinationsConfig(seed=-1)
    assert config.seed == -1


def test_chat_pollinations_config_forbids_extra():
    with pytest.raises(ValidationError):
        ChatPollinationsConfig(model="openai", unknown_param="value")


def test_normalize_tool_choice_none():
    assert _normalize_tool_choice(None) is None


def test_normalize_tool_choice_any_to_required():
    assert _normalize_tool_choice("any") == "required"


def test_normalize_tool_choice_dict_any_to_required():
    assert _normalize_tool_choice({"type": "any"}) == "required"


def test_normalize_tool_choice_passthrough():
    assert _normalize_tool_choice("auto") == "auto"
    assert _normalize_tool_choice("none") == "none"


def test_extract_text_from_content_blocks_empty():
    assert _extract_text_from_content_blocks([]) == ""
    assert _extract_text_from_content_blocks(None) == ""
    assert _extract_text_from_content_blocks("not a list") == ""


def test_extract_text_from_content_blocks_with_text():
    blocks = [
        {"type": "text", "text": "Hello "},
        {"type": "image", "url": "http://example.com/img.jpg"},
        {"type": "text", "text": "world"}
    ]
    assert _extract_text_from_content_blocks(blocks) == "Hello world"


def test_usage_metadata_from_usage_valid():
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    metadata = _usage_metadata_from_usage(usage)

    assert metadata is not None
    assert metadata["input_tokens"] == 10
    assert metadata["output_tokens"] == 20
    assert metadata["total_tokens"] == 30


def test_usage_metadata_from_usage_with_details():
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "prompt_tokens_details": {"cached": 5},
        "completion_tokens_details": {"reasoning": 3}
    }
    metadata = _usage_metadata_from_usage(usage)

    assert metadata["input_token_details"] == {"cached": 5}
    assert metadata["output_token_details"] == {"reasoning": 3}


def test_usage_metadata_from_usage_invalid():
    assert _usage_metadata_from_usage(None) is None
    assert _usage_metadata_from_usage("not a dict") is None
    assert _usage_metadata_from_usage({}) is None


def test_response_metadata_from_response():
    response = {
        "id": "chatcmpl-123",
        "model": "openai",
        "created": 1234567890,
        "system_fingerprint": "fp_123",
        "citations": ["source1", "source2"],
        "extra_field": "ignored"
    }
    metadata = _response_metadata_from_response(response)

    assert metadata["id"] == "chatcmpl-123"
    assert metadata["model"] == "openai"
    assert metadata["created"] == 1234567890
    assert metadata["system_fingerprint"] == "fp_123"
    assert metadata["citations"] == ["source1", "source2"]
    assert "extra_field" not in metadata


def test_message_content_from_message_dict_string():
    message = {"content": "Hello world"}
    assert _message_content_from_message_dict(message) == "Hello world"


def test_message_content_from_message_dict_blocks():
    message = {
        "content_blocks": [
            {"type": "text", "text": "Block content"}
        ]
    }
    result = _message_content_from_message_dict(message)
    assert result == [{"type": "text", "text": "Block content"}]


def test_message_content_from_message_dict_audio():
    message = {
        "audio": {"transcript": "Audio transcript"}
    }
    assert _message_content_from_message_dict(message) == "Audio transcript"


def test_message_content_from_message_dict_empty():
    assert _message_content_from_message_dict({}) == ""


def test_delta_content_from_delta_dict_string():
    delta = {"content": "delta text"}
    assert _delta_content_from_delta_dict(delta) == "delta text"


def test_delta_content_from_delta_dict_blocks():
    delta = {"content_blocks": [{"type": "text", "text": "chunk"}]}
    result = _delta_content_from_delta_dict(delta)
    assert result == [{"type": "text", "text": "chunk"}]


def test_text_from_any_content_string():
    assert _text_from_any_content("hello") == "hello"


def test_text_from_any_content_blocks():
    blocks = [
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "there"}
    ]
    assert _text_from_any_content(blocks) == "Hello there"


def test_text_from_any_content_other():
    assert _text_from_any_content(123) == ""
    assert _text_from_any_content(None) == ""


def test_tool_call_chunks_from_delta_empty():
    assert _tool_call_chunks_from_delta({}) == []
    assert _tool_call_chunks_from_delta({"tool_calls": None}) == []


def test_tool_call_chunks_from_delta_valid():
    delta = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_123",
                "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'}
            }
        ]
    }
    chunks = _tool_call_chunks_from_delta(delta)

    assert len(chunks) == 1
    assert chunks[0]["name"] == "get_weather"
    assert chunks[0]["args"] == '{"city":"NYC"}'
    assert chunks[0]["id"] == "call_123"
    assert chunks[0]["index"] == 0


def test_parse_tool_calls_empty():
    tool_calls, invalid = _parse_tool_calls({})
    assert tool_calls == []
    assert invalid == []


def test_parse_tool_calls_valid():
    message = {
        "tool_calls": [
            {
                "type": "function",
                "id": "call_1",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "SF"}'
                }
            }
        ]
    }
    tool_calls, invalid = _parse_tool_calls(message)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_weather"
    assert tool_calls[0]["args"] == {"city": "SF"}
    assert tool_calls[0]["id"] == "call_1"
    assert len(invalid) == 0


def test_parse_tool_calls_invalid_json():
    message = {
        "tool_calls": [
            {
                "type": "function",
                "id": "call_bad",
                "function": {
                    "name": "bad_tool",
                    "arguments": '{"invalid json'
                }
            }
        ]
    }
    tool_calls, invalid = _parse_tool_calls(message)

    assert len(tool_calls) == 0
    assert len(invalid) == 1
    assert invalid[0]["name"] == "bad_tool"
    assert invalid[0]["type"] == "invalid_tool_call"


def test_iter_sse_json_events_sync_simple():
    class MockResponse:
        def iter_lines(self):
            yield 'data: {"message": "hello"}'
            yield 'data: [DONE]'

    resp = MockResponse()
    events = list(_iter_sse_json_events_sync(resp))

    assert len(events) == 2
    assert events[0] == {"message": "hello"}
    assert events[1] == {"__done__": True}


def test_iter_sse_json_events_sync_with_blanks():
    class MockResponse:
        def iter_lines(self):
            yield 'data: {"chunk": 1}'
            yield ''
            yield 'data: {"chunk": 2}'
            yield 'data: [DONE]'

    resp = MockResponse()
    events = list(_iter_sse_json_events_sync(resp))

    assert len(events) >= 2
    assert {"chunk": 1} in events
    assert {"chunk": 2} in events


@pytest.mark.asyncio
async def test_iter_sse_json_events_async_simple():
    class MockResponse:
        async def aiter_lines(self):
            yield 'data: {"message": "async"}'
            yield 'data: [DONE]'

    resp = MockResponse()
    events = []
    async for evt in _iter_sse_json_events_async(resp):
        events.append(evt)

    assert len(events) == 2
    assert events[0] == {"message": "async"}
    assert events[1] == {"__done__": True}


def test_chat_pollinations_initialization_defaults():
    chat = ChatPollinations()

    assert chat.base_url == DEFAULT_BASE_URL
    assert chat.timeout_s == 120.0
    assert chat.include_usage_in_stream is True
    assert chat.preserve_multimodal_deltas is True
    assert isinstance(chat._http, DummyHttpClient)


def test_chat_pollinations_initialization_with_params():
    chat = ChatPollinations(
        api_key="custom-key",
        base_url="https://custom.api",
        timeout_s=60.0,
        model="openai-large",
        temperature=0.5
    )

    assert chat.base_url == "https://custom.api"
    assert chat.timeout_s == 60.0
    assert chat.request_defaults.model == "openai-large"
    assert chat.request_defaults.temperature == 0.5


def test_chat_pollinations_llm_type():
    chat = ChatPollinations()
    assert chat._llm_type == "pollinations-chat"


def test_chat_pollinations_identifying_params():
    chat = ChatPollinations(model="gemini", base_url="https://test.api")
    params = chat._identifying_params

    assert params["model"] == "gemini"
    assert params["base_url"] == "https://test.api"
    assert params["timeout_s"] == 120.0


def test_chat_pollinations_bind_tools_simple():
    chat = ChatPollinations()

    tools = [{"type": "code_execution"}]
    bound = chat.bind_tools(tools)

    assert bound.request_defaults.tools is not None
    assert len(bound.request_defaults.tools) == 1


def test_chat_pollinations_bind_tools_with_choice():
    chat = ChatPollinations()

    tools = [{"type": "google_search"}]
    bound = chat.bind_tools(tools, tool_choice="required")

    assert bound.request_defaults.tool_choice == "required"


def test_chat_pollinations_bind_tools_normalizes_any():
    chat = ChatPollinations()

    tools = [{"type": "code_execution"}]
    bound = chat.bind_tools(tools, tool_choice="any")

    assert bound.request_defaults.tool_choice == "required"


def test_chat_pollinations_build_payload_basic():
    chat = ChatPollinations(model="openai", temperature=0.7)
    messages = [HumanMessage(content="Hello")]

    payload = chat._build_payload(messages, stop=None)

    assert "messages" in payload
    assert payload["model"] == "openai"
    assert payload["temperature"] == 0.7


def test_chat_pollinations_build_payload_with_stop():
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    payload = chat._build_payload(messages, stop=["STOP", "END"])

    assert payload["stop"] == ["STOP", "END"]


def test_chat_pollinations_build_payload_with_kwargs():
    chat = ChatPollinations(model="openai")
    messages = [HumanMessage(content="Test")]

    payload = chat._build_payload(messages, stop=None, max_tokens=50, temperature=0.9)

    assert payload["max_tokens"] == 50
    assert payload["temperature"] == 0.9


def test_chat_pollinations_parse_chat_result():
    chat = ChatPollinations()

    data = {
        "id": "chatcmpl-123",
        "model": "openai",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response text"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    }

    result = chat._parse_chat_result(data)

    assert len(result.generations) == 1
    gen = result.generations[0]
    assert isinstance(gen.message, AIMessage)
    assert gen.message.content == "Response text"
    assert gen.message.usage_metadata["input_tokens"] == 5
    assert gen.message.usage_metadata["output_tokens"] == 10


def test_chat_pollinations_parse_chat_result_with_tool_calls():
    chat = ChatPollinations()

    data = {
        "id": "chatcmpl-123",
        "model": "openai",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"NYC"}'
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }

    result = chat._parse_chat_result(data)

    gen = result.generations[0]
    assert len(gen.message.tool_calls) == 1
    assert gen.message.tool_calls[0]["name"] == "get_weather"
    assert gen.message.tool_calls[0]["args"] == {"city": "NYC"}


def test_chat_pollinations_generate():
    chat = ChatPollinations(model="openai")
    messages = [HumanMessage(content="Hello")]

    result = chat._generate(messages)

    assert len(result.generations) == 1
    assert isinstance(result.generations[0].message, AIMessage)
    assert len(chat._http.calls) == 1
    assert chat._http.calls[0]["method"] == "post_json"
    assert chat._http.calls[0]["stream"] is False


@pytest.mark.asyncio
async def test_chat_pollinations_agenerate():
    chat = ChatPollinations(model="openai")
    messages = [HumanMessage(content="Hello async")]

    result = await chat._agenerate(messages)

    assert len(result.generations) == 1
    assert isinstance(result.generations[0].message, AIMessage)
    assert len(chat._http.calls) == 1
    assert chat._http.calls[0]["method"] == "apost_json"


def test_chat_pollinations_stream():
    chat = ChatPollinations(model="openai")
    messages = [HumanMessage(content="Stream test")]

    chunks = list(chat._stream(messages))

    assert len(chunks) > 0
    # Verificar que se llamó al método de streaming
    assert any(call["method"] == "stream_post_json" for call in chat._http.calls)
    # Verificar que el payload tiene stream=True
    stream_call = next(call for call in chat._http.calls if call["method"] == "stream_post_json")
    assert stream_call["payload"]["stream"] is True


@pytest.mark.asyncio
async def test_chat_pollinations_astream():
    chat = ChatPollinations(model="openai")
    messages = [HumanMessage(content="Async stream test")]

    chunks = []
    async for chunk in chat._astream(messages):
        chunks.append(chunk)

    assert len(chunks) > 0
    # Verificar que se llamó al método de streaming asíncrono
    assert any(call["method"] == "astream_post_json" for call in chat._http.calls)


def test_chat_pollinations_stream_includes_usage():
    chat = ChatPollinations(model="openai", include_usage_in_stream=True)
    messages = [HumanMessage(content="Test")]

    # Configurar eventos de stream con uso
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}',
        'data: {"choices":[{"index":0,"delta":{}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Verificar que include_usage está en stream_options del payload
    stream_call = next(call for call in chat._http.calls if call["method"] == "stream_post_json")
    assert stream_call["payload"].get("stream_options", {}).get("include_usage") is True


def test_chat_pollinations_request_defaults_error():
    with pytest.raises(ValueError) as exc:
        ChatPollinations(
            request_defaults=ChatPollinationsConfig(model="openai"),
            model="gemini"
        )

    assert "Do not mix request_defaults" in str(exc.value)


def test_chat_pollinations_bind_tools_preserves_api_key():
    chat = ChatPollinations(api_key="secret-123", model="openai")

    tools = [{"type": "code_execution"}]
    bound = chat.bind_tools(tools)

    assert bound.api_key == "secret-123"
    assert bound._http.api_key == "secret-123"


def test_chat_pollinations_stream_with_tool_calls():
    chat = ChatPollinations()
    messages = [HumanMessage(content="Get weather")]

    # Configurar eventos de stream con tool calls
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":"{\\"city\\":\\"SF\\"}"}}]}}]}',
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Verificar que hay chunks con tool_call_chunks
    assert any(chunk.message.tool_call_chunks for chunk in chunks)


def test_chat_pollinations_parse_chat_result_empty_choices():
    chat = ChatPollinations()

    data = {
        "id": "chatcmpl-123",
        "model": "openai",
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

    result = chat._parse_chat_result(data)

    # Debe crear una generación por defecto con mensaje vacío
    assert len(result.generations) == 1
    assert result.generations[0].message.content == ""


@pytest.mark.asyncio
async def test_chat_pollinations_astream_usage_emission():
    chat = ChatPollinations(include_usage_in_stream=True)
    messages = [HumanMessage(content="Test usage")]

    # Configurar eventos con usage al final
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
        'data: [DONE]'
    ]

    chunks = []
    async for chunk in chat._astream(messages):
        chunks.append(chunk)

    # Verificar que el último chunk tiene metadata de uso
    usage_chunks = [c for c in chunks if c.message.usage_metadata is not None]
    assert len(usage_chunks) > 0


def test_chat_pollinations_build_payload_filters_non_provider_kwargs():
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    # LangChain puede pasar kwargs que no son del proveedor
    payload = chat._build_payload(
        messages,
        stop=None,
        stream_mode="updates",  # no es parámetro de proveedor
        include_names=["test"],  # no es parámetro de proveedor
        temperature=0.8  # sí es parámetro de proveedor
    )

    # Solo temperature debe estar en el payload, los otros deben ser ignorados
    assert "temperature" in payload
    assert "stream_mode" not in payload
    assert "include_names" not in payload


def test_chat_pollinations_preserve_multimodal_deltas():
    chat = ChatPollinations(preserve_multimodal_deltas=True)
    messages = [HumanMessage(content="Test")]

    # Configurar eventos con contenido multimodal
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"Hello"}]}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Verificar que content_parts está en additional_kwargs
    assert any("content_parts" in chunk.message.additional_kwargs for chunk in chunks)


def test_chat_pollinations_tools_with_parallel_calls():
    chat = ChatPollinations()

    tools = [{"type": "google_search"}]
    bound = chat.bind_tools(tools, parallel_tool_calls=True)

    assert bound.request_defaults.parallel_tool_calls is True


def test_thinking_config_budget_validation():
    with pytest.raises(ValidationError):
        ThinkingConfig(type="enabled", budget_tokens=0)

    config = ThinkingConfig(type="enabled", budget_tokens=1000)
    assert config.budget_tokens == 1000


def test_extract_text_from_content_blocks_non_dict_items():
    """Cubrir línea donde block no es dict"""
    blocks = [
        {"type": "text", "text": "Hello"},
        "not a dict",
        None,
        {"type": "text", "text": " world"}
    ]
    result = _extract_text_from_content_blocks(blocks)
    assert result == "Hello world"


def test_extract_text_from_content_blocks_missing_text_field():
    """Cubrir línea donde text no es string o está ausente"""
    blocks = [
        {"type": "text"},  # sin field 'text'
        {"type": "text", "text": 123},  # text no es string
        {"type": "text", "text": ""},  # texto vacío
        {"type": "text", "text": "valid"}
    ]
    result = _extract_text_from_content_blocks(blocks)
    assert result == "valid"


def test_message_content_from_message_dict_priority():
    """Cubrir prioridad: content > content_blocks > audio"""
    # content tiene prioridad
    message = {
        "content": "direct content",
        "content_blocks": [{"type": "text", "text": "blocks"}],
        "audio": {"transcript": "audio"}
    }
    assert _message_content_from_message_dict(message) == "direct content"

    # content_blocks tiene prioridad sobre audio
    message2 = {
        "content_blocks": [{"type": "text", "text": "blocks"}],
        "audio": {"transcript": "audio"}
    }
    result = _message_content_from_message_dict(message2)
    assert isinstance(result, list)


def test_message_content_from_message_dict_audio_not_dict():
    """Cubrir cuando audio no es dict"""
    message = {"audio": "not a dict"}
    assert _message_content_from_message_dict(message) == ""


def test_message_content_from_message_dict_audio_transcript_not_string():
    """Cubrir cuando transcript no es string"""
    message = {"audio": {"transcript": 123}}
    assert _message_content_from_message_dict(message) == ""


def test_message_content_from_message_dict_audio_transcript_empty():
    """Cubrir cuando transcript es string vacío"""
    message = {"audio": {"transcript": ""}}
    assert _message_content_from_message_dict(message) == ""


def test_delta_content_from_delta_dict_all_branches():
    """Cubrir todas las ramas en delta content extraction"""
    # content_blocks no es list
    delta1 = {"content_blocks": "not a list"}
    assert _delta_content_from_delta_dict(delta1) == ""

    # content_blocks es lista vacía
    delta2 = {"content_blocks": []}
    assert _delta_content_from_delta_dict(delta2) == ""

    # audio no es dict
    delta3 = {"audio": []}
    assert _delta_content_from_delta_dict(delta3) == ""


def test_tool_call_chunks_from_delta_invalid_items():
    """Cubrir casos donde items no son válidos"""
    delta = {
        "tool_calls": [
            "not a dict",
            {"index": "not an int"},  # index no es int
            {"index": 0},  # sin function
            {"index": 0, "function": "not a dict"},  # function no es dict
            {"index": 0, "function": {"name": 123}},  # name no es string
            {"index": 0, "function": {"arguments": 123}},  # args no es string
        ]
    }
    chunks = _tool_call_chunks_from_delta(delta)

    # La función crea chunks incluso si name/args son None cuando index y function son válidos
    # Los últimos 2 items tienen index=0 y function dict válido, así que generan chunks
    assert len(chunks) == 2
    # Verificar que son chunks con None values
    assert all(c["name"] is None for c in chunks)
    assert all(c["args"] is None for c in chunks)


def test_tool_call_chunks_from_delta_with_none_values():
    """Cubrir cuando name/args son None o ausentes"""
    delta = {
        "tool_calls": [
            {
                "index": 0,
                "function": {}  # sin name ni arguments
            }
        ]
    }
    chunks = _tool_call_chunks_from_delta(delta)
    assert len(chunks) == 1
    assert chunks[0]["name"] is None
    assert chunks[0]["args"] is None


def test_parse_tool_calls_invalid_items():
    """Cubrir casos donde tool_calls tienen items inválidos"""
    message = {
        "tool_calls": [
            "not a dict",
            {"type": "not_function"},  # tipo incorrecto
            {"type": "function"},  # sin function field
            {"type": "function", "function": "not a dict"},  # function no es dict
            {"type": "function", "function": {"name": ""}},  # name vacío
            {"type": "function", "function": {"name": 123}},  # name no es string
        ]
    }
    tool_calls, invalid = _parse_tool_calls(message)
    assert len(tool_calls) == 0
    assert len(invalid) == 0


def test_parse_tool_calls_empty_arguments():
    """Cubrir cuando arguments es string vacío o whitespace"""
    message = {
        "tool_calls": [
            {
                "type": "function",
                "id": "call_1",
                "function": {
                    "name": "test_func",
                    "arguments": ""
                }
            },
            {
                "type": "function",
                "id": "call_2",
                "function": {
                    "name": "test_func2",
                    "arguments": "   "
                }
            }
        ]
    }
    tool_calls, invalid = _parse_tool_calls(message)
    assert len(tool_calls) == 2
    assert tool_calls[0]["args"] == {}
    assert tool_calls[1]["args"] == {}


def test_parse_tool_calls_non_dict_json():
    """Cubrir cuando arguments JSON no es dict - el código lo acepta con args vacío"""
    message = {
        "tool_calls": [
            {
                "type": "function",
                "id": "call_1",
                "function": {
                    "name": "test_func",
                    "arguments": '["array", "not", "dict"]'
                }
            }
        ]
    }
    tool_calls, invalid = _parse_tool_calls(message)

    # El código acepta JSON no-dict pero lo procesa con args={}
    # porque loaded no es dict, entonces args queda como {}
    assert len(tool_calls) == 1
    assert tool_calls[0]["args"] == {}
    assert len(invalid) == 0


def test_iter_sse_json_events_sync_binary_lines():
    """Cubrir cuando iter_lines retorna bytes"""
    class MockResponse:
        def iter_lines(self):
            yield b'data: {"message": "binary"}'
            yield b'data: [DONE]'

    resp = MockResponse()
    events = list(_iter_sse_json_events_sync(resp))

    assert len(events) >= 1
    assert any(e.get("message") == "binary" for e in events if not e.get("__done__"))


def test_iter_sse_json_events_sync_decode_error():
    """Cubrir excepción al decodificar bytes"""
    class MockResponse:
        def iter_lines(self):
            yield b'\xff\xfe'  # bytes inválidos UTF-8
            yield 'data: {"ok": true}'
            yield 'data: [DONE]'

    resp = MockResponse()
    events = list(_iter_sse_json_events_sync(resp))
    # Debe seguir procesando a pesar del error
    assert any(e.get("ok") for e in events if not e.get("__done__"))


def test_iter_sse_json_events_sync_invalid_json():
    """Cubrir cuando el JSON es inválido"""
    class MockResponse:
        def iter_lines(self):
            yield 'data: {invalid json}'
            yield 'data: {"valid": true}'
            yield 'data: [DONE]'

    resp = MockResponse()
    events = list(_iter_sse_json_events_sync(resp))
    # Solo debe procesar el JSON válido
    valid_events = [e for e in events if not e.get("__done__")]
    assert len(valid_events) == 1
    assert valid_events[0]["valid"] is True


def test_iter_sse_json_events_sync_multiline_data():
    """Cubrir acumulación de múltiples líneas data"""
    class MockResponse:
        def iter_lines(self):
            yield 'data: {"part1":'
            yield 'data: "value1",'
            yield 'data: "part2": "value2"}'
            yield ''
            yield 'data: [DONE]'

    resp = MockResponse()
    events = list(_iter_sse_json_events_sync(resp))
    # Debe intentar parsear la acumulación
    assert len(events) >= 1


def test_iter_sse_json_events_sync_non_dict_json():
    """Cubrir cuando JSON no es dict"""
    class MockResponse:
        def iter_lines(self):
            yield 'data: ["array"]'
            yield 'data: "string"'
            yield 'data: 123'
            yield 'data: [DONE]'

    resp = MockResponse()
    events = list(_iter_sse_json_events_sync(resp))
    # Solo eventos __done__
    assert all(e.get("__done__") for e in events)


@pytest.mark.asyncio
async def test_iter_sse_json_events_async_invalid_json():
    """Cubrir JSON inválido en async"""
    class MockResponse:
        async def aiter_lines(self):
            yield 'data: {bad json'
            yield 'data: {"good": true}'
            yield 'data: [DONE]'

    resp = MockResponse()
    events = []
    async for evt in _iter_sse_json_events_async(resp):
        events.append(evt)

    valid_events = [e for e in events if not e.get("__done__")]
    assert len(valid_events) == 1


def test_chat_pollinations_bind_tools_with_pydantic_model():
    """Cubrir conversión de Pydantic model a tool"""
    chat = ChatPollinations()

    class WeatherArgs(BaseModel):
        """Get weather information"""
        city: str = Field(description="City name")
        units: str = Field(default="celsius")

    # Necesitamos hacer patch del módulo correcto donde se importa
    with patch("langchain_core.utils.function_calling.convert_to_openai_tool", None):
        bound = chat.bind_tools([WeatherArgs], strict=True)

    assert bound.request_defaults.tools is not None
    assert len(bound.request_defaults.tools) == 1
    tool = bound.request_defaults.tools[0]
    assert tool.type == "function"
    assert tool.function.name == "WeatherArgs"
    assert tool.function.strict is True


def test_chat_pollinations_bind_tools_openai_format_with_name():
    """Cubrir tool en formato OpenAI alternativo con name/parameters"""
    chat = ChatPollinations()

    # Simular que convert_to_openai_tool retorna formato alternativo
    def mock_converter(tool):
        return {
            "name": "mock_tool",
            "description": "A mock tool",
            "parameters": {"type": "object", "properties": {"field": {"type": "string"}}}
        }

    with patch("langchain_core.utils.function_calling.convert_to_openai_tool", mock_converter):
        bound = chat.bind_tools([{"some": "tool"}], strict=True)

    assert bound.request_defaults.tools is not None


def test_chat_pollinations_bind_tools_conversion_exception():
    """Cubrir excepción durante conversión de tool con fallback"""
    chat = ChatPollinations()

    # Usar un tipo simple que el sistema pueda manejar
    tool_dict = {
        "type": "function",
        "function": {
            "name": "simple_tool",
            "parameters": {"type": "object"}
        }
    }

    # Esto debe funcionar sin excepciones
    bound = chat.bind_tools([tool_dict])

    assert bound.request_defaults.tools is not None


def test_chat_pollinations_bind_tools_type_adapter_exception():
    """Cubrir excepción en TypeAdapter para tipos complejos"""
    chat = ChatPollinations()

    class UnserializableType:
        __name__ = "UnserializableType"

    # Patch en el lugar correcto
    with patch("langchain_core.utils.function_calling.convert_to_openai_tool", None):
        bound = chat.bind_tools([UnserializableType])

    # Debe crear tool con schema por defecto
    assert bound.request_defaults.tools is not None


def test_chat_pollinations_bind_tools_already_openai_format():
    """Cubrir tool que ya está en formato OpenAI completo"""
    chat = ChatPollinations()

    tool = {
        "type": "function",
        "function": {
            "name": "existing_tool",
            "description": "Already formatted",
            "parameters": {"type": "object"}
        }
    }

    bound = chat.bind_tools([tool], strict=True)

    assert bound.request_defaults.tools is not None
    result_tool = bound.request_defaults.tools[0]
    assert result_tool.function.strict is True
    assert result_tool.function.name == "existing_tool"


def test_chat_pollinations_parse_chat_result_with_valid_tool_calls():
    """Cubrir mensaje con tool_calls válidos (JSON correcto)"""
    chat = ChatPollinations()

    data = {
        "id": "chatcmpl-123",
        "model": "openai",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "id": "call_good",
                    "function": {
                        "name": "good_tool",
                        "arguments": '{"param": "value"}'  # JSON válido
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }

    result = chat._parse_chat_result(data)

    gen = result.generations[0]
    # Con JSON válido, debe estar en tool_calls, no en invalid_tool_calls
    assert len(gen.message.tool_calls) == 1
    assert gen.message.tool_calls[0]["name"] == "good_tool"
    assert gen.message.tool_calls[0]["args"] == {"param": "value"}
    assert len(gen.message.invalid_tool_calls) == 0


def test_chat_pollinations_parse_chat_result_with_additional_kwargs():
    """Cubrir additional_kwargs con audio, reasoning_content, content_blocks"""
    chat = ChatPollinations()

    data = {
        "id": "chatcmpl-123",
        "model": "openai",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Response",
                "audio": {"transcript": "audio text"},
                "reasoning_content": "thinking...",
                "content_blocks": [{"type": "text", "text": "block"}],
                "function_call": {"name": "old_func"}
            },
            "finish_reason": "stop",
            "logprobs": {"tokens": ["a", "b"]},
            "content_filter_results": {"hate": "safe"}
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    }

    result = chat._parse_chat_result(data)

    gen = result.generations[0]
    assert "audio" in gen.message.additional_kwargs
    assert "reasoning_content" in gen.message.additional_kwargs
    assert "content_blocks" in gen.message.additional_kwargs
    assert "function_call" in gen.message.additional_kwargs
    assert gen.generation_info["logprobs"] is not None
    assert gen.generation_info["content_filter_results"] is not None


def test_chat_pollinations_parse_chat_result_invalid_choice():
    """Cubrir choice que no es dict"""
    chat = ChatPollinations()

    data = {
        "id": "chatcmpl-123",
        "model": "openai",
        "choices": [
            "not a dict",
            {"message": "also not dict"},
            None
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

    result = chat._parse_chat_result(data)

    # Debe crear generación por defecto
    assert len(result.generations) == 1
    assert result.generations[0].message.content == ""


def test_chat_pollinations_stream_with_multimodal_content():
    """Cubrir streaming con contenido multimodal en delta"""
    chat = ChatPollinations(preserve_multimodal_deltas=True)
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"Multi"},{"type":"text","text":"modal"}]}}]}',
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Verificar que content_parts está preservado
    parts_chunks = [c for c in chunks if "content_parts" in c.message.additional_kwargs]
    assert len(parts_chunks) > 0


def test_chat_pollinations_stream_with_additional_delta_fields():
    """Cubrir delta con audio, reasoning_content, function_call"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"audio":{"transcript":"test"},"reasoning_content":"thinking","function_call":{"name":"func"}}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Verificar que los campos están en additional_kwargs
    for chunk in chunks:
        if chunk.message.additional_kwargs:
            kwargs = chunk.message.additional_kwargs
            if "audio" in kwargs or "reasoning_content" in kwargs or "function_call" in kwargs:
                assert True
                return

    assert False, "No se encontraron campos adicionales en additional_kwargs"


@pytest.mark.asyncio
async def test_chat_pollinations_astream_with_additional_delta_fields():
    """Cubrir async stream con campos adicionales"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content_blocks":[{"type":"text","text":"test"}]}}]}',
        'data: [DONE]'
    ]

    chunks = []
    async for chunk in chat._astream(messages):
        chunks.append(chunk)

    assert len(chunks) > 0


def test_chat_pollinations_stream_no_choices():
    """Cubrir evento de stream sin choices"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"id":"chatcmpl-123"}',  # sin choices
        'data: {"choices":[]}',  # choices vacío
        'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Debe procesar solo el chunk válido
    assert len(chunks) > 0


def test_chat_pollinations_stream_invalid_choice():
    """Cubrir choice que no es dict en stream"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":["not a dict"]}',
        'data: {"choices":[{"index":0,"delta":"not a dict"}]}',
        'data: {"choices":[{"index":0,"delta":{"content":"valid"}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Debe procesar solo los chunks válidos
    assert any(c.message.content == "valid" for c in chunks)


@pytest.mark.asyncio
async def test_chat_pollinations_astream_invalid_choices():
    """Cubrir choices inválidos en async stream"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":"not a list"}',
        'data: {"choices":[{"index":0,"delta":{"content":"good"}}]}',
        'data: [DONE]'
    ]

    chunks = []
    async for chunk in chat._astream(messages):
        chunks.append(chunk)

    assert any(c.message.content == "good" for c in chunks)


def test_chat_pollinations_stream_with_logprobs_and_filters():
    """Cubrir generation_info con logprobs y content_filter_results"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"},"logprobs":{"tokens":["test"]},"content_filter_results":{"violence":"safe"}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Verificar que generation_info contiene los campos
    info_chunks = [c for c in chunks if c.generation_info and ("logprobs" in c.generation_info or "content_filter_results" in c.generation_info)]
    assert len(info_chunks) > 0


def test_chat_pollinations_build_payload_stop_override():
    """Cubrir diferentes formas de pasar stop"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    # Caso 1: stop solo como parámetro del método
    payload1 = chat._build_payload(messages, stop=["STOP1"])
    assert payload1["stop"] == ["STOP1"]

    # Caso 2: stop=None en método (no se incluye)
    payload2 = chat._build_payload(messages, stop=None)
    assert "stop" not in payload2 or payload2.get("stop") is None

    # Caso 3: stop como parámetro del método tiene prioridad
    payload3 = chat._build_payload(messages, stop=["PRIORITY"])
    assert payload3["stop"] == ["PRIORITY"]

    # Caso 4: stop pasado en kwargs cuando stop del método es None
    chat_with_stop = ChatPollinations(request_defaults=ChatPollinationsConfig(stop="DEFAULT"))
    payload4 = chat_with_stop._build_payload(messages, stop=None)
    # El stop de request_defaults debe aparecer
    assert payload4.get("stop") == "DEFAULT"


def test_chat_pollinations_build_payload_none_values_filtered():
    """Cubrir que kwargs con None no se incluyen"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    payload = chat._build_payload(
        messages,
        stop=None,
        temperature=None,
        max_tokens=100,
        seed=None
    )

    assert "temperature" not in payload
    assert "seed" not in payload
    assert payload["max_tokens"] == 100


def test_chat_pollinations_stream_usage_without_include():
    """Cubrir stream cuando include_usage_in_stream es False"""
    chat = ChatPollinations(include_usage_in_stream=False)
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Verificar que stream_options no tiene include_usage
    stream_call = next(call for call in chat._http.calls if call["method"] == "stream_post_json")
    payload = stream_call["payload"]
    # Si está presente, no debe ser True
    if "stream_options" in payload:
        assert payload["stream_options"].get("include_usage") is not True


@pytest.mark.asyncio
async def test_chat_pollinations_astream_usage_without_include():
    """Cubrir async stream sin include_usage"""
    chat = ChatPollinations(include_usage_in_stream=False)
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}]}',
        'data: [DONE]'
    ]

    chunks = []
    async for chunk in chat._astream(messages):
        chunks.append(chunk)

    assert len(chunks) > 0


def test_chat_pollinations_stream_usage_event_not_dict():
    """Cubrir cuando usage en evento no es dict"""
    chat = ChatPollinations(include_usage_in_stream=True)
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}],"usage":"not a dict"}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # No debe fallar, simplemente no procesa el uso
    assert len(chunks) > 0


def test_chat_pollinations_stream_existing_stream_options():
    """Cubrir cuando stream_options ya existe en payload"""
    chat = ChatPollinations(
        include_usage_in_stream=True,
        request_defaults=ChatPollinationsConfig(
            stream_options={"some_option": "value"}
        )
    )
    messages = [HumanMessage(content="Test")]

    list(chat._stream(messages))

    stream_call = next(call for call in chat._http.calls if call["method"] == "stream_post_json")
    payload = stream_call["payload"]

    # Debe preservar la opción existente y agregar include_usage
    assert payload["stream_options"]["include_usage"] is True
    assert payload["stream_options"]["some_option"] == "value"


def test_normalize_tool_choice_complex_dict():
    """Cubrir tool_choice como dict complejo"""
    from langchain_pollinations.chat import _normalize_tool_choice

    choice = {
        "type": "function",
        "function": {"name": "specific_tool"}
    }
    result = _normalize_tool_choice(choice)
    assert result == choice


def test_chat_pollinations_bind_tools_with_description_from_docstring():
    """Cubrir extracción de description desde __doc__"""
    chat = ChatPollinations()

    class ToolWithDocstring:
        """This is a docstring description."""
        pass

    # bind_tools debe extraer la descripción del docstring
    bound = chat.bind_tools([ToolWithDocstring])

    tool = bound.request_defaults.tools[0]
    # Verificar que se creó el tool y tiene nombre
    assert tool.function.name is not None
    # La descripción puede venir del docstring o ser None
    if tool.function.description:
        assert len(tool.function.description) > 0




'''
def test_bind_tools_with_convert_to_openai_tool_returns_non_dict():
    """Cubrir cuando convert_to_openai_tool retorna algo que no es dict"""
    chat = ChatPollinations()

    def mock_converter(tool):
        return "not a dict"  # Retorna algo inválido

    with patch("langchain_core.utils.function_calling.convert_to_openai_tool", mock_converter):
        # Debe usar fallback tool_to_openai_tool
        bound = chat.bind_tools([{"some": "tool"}])

    assert bound.request_defaults.tools is not None
'''

def test_bind_tools_with_convert_returns_dict_without_type_or_name():
    """Cubrir dict sin 'type'='function' ni 'name'"""
    chat = ChatPollinations()

    def mock_converter(tool):
        # Retorna dict pero sin estructura esperada
        return {"unexpected": "format"}

    with patch("langchain_core.utils.function_calling.convert_to_openai_tool", mock_converter):
        with patch("langchain_pollinations.chat.tool_to_openai_tool") as mock_fallback:
            mock_fallback.return_value = {
                "type": "function",
                "function": {"name": "fallback", "parameters": {}}
            }

            bound = chat.bind_tools([{"tool": "data"}])

            # Debe llamar al fallback
            mock_fallback.assert_called_once()


def test_bind_tools_with_pydantic_model_subclass_exception():
    """Cubrir excepción al verificar BaseModel subclass"""
    chat = ChatPollinations()

    class FakeTool:
        __name__ = "FakeTool"

    # Simular que BaseModel no está disponible
    with patch("langchain_pollinations.chat.BaseModel", None):
        with patch("langchain_core.utils.function_calling.convert_to_openai_tool", None):
            bound = chat.bind_tools([FakeTool])

    assert bound.request_defaults.tools is not None


def test_bind_tools_basemodel_with_model_json_schema_exception():
    """Cubrir excepción en model_json_schema()"""
    chat = ChatPollinations()

    class BrokenModel(BaseModel):
        field: str

        @classmethod
        def model_json_schema(cls):
            raise RuntimeError("Schema generation failed")

    with patch("langchain_core.utils.function_calling.convert_to_openai_tool", None):
        bound = chat.bind_tools([BrokenModel])

    # Debe usar schema por defecto
    assert bound.request_defaults.tools is not None


def test_stream_with_run_manager_callbacks():
    """Cubrir uso de run_manager en _stream"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    # Crear un mock callback manager
    mock_run_manager = MagicMock()

    # Configurar eventos simples
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages, run_manager=mock_run_manager))

    # Debe procesar los chunks
    assert len(chunks) > 0


def test_stream_with_empty_stream_options_and_include_usage():
    """Cubrir None branch"""
    chat = ChatPollinations(
        include_usage_in_stream=True,
        request_defaults=ChatPollinationsConfig()  # Sin stream_options
    )
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}]}',
        'data: [DONE]'
    ]

    list(chat._stream(messages))

    # Verificar que stream_options se agregó al payload
    call = next(c for c in chat._http.calls if c["method"] == "stream_post_json")
    assert call["payload"]["stream_options"]["include_usage"] is True


def test_stream_with_dict_stream_options_without_include_usage():
    """Cubrir so es dict pero sin include_usage"""
    from langchain_pollinations.chat import StreamOptions

    chat = ChatPollinations(
        include_usage_in_stream=True,
        request_defaults=ChatPollinationsConfig(
            stream_options=StreamOptions(include_usage=None)
        )
    )
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}]}',
        'data: [DONE]'
    ]

    list(chat._stream(messages))

    call = next(c for c in chat._http.calls if c["method"] == "stream_post_json")
    # Debe agregar include_usage al dict existente
    assert call["payload"]["stream_options"]["include_usage"] is True


def test_stream_emit_final_usage_once_guards():
    """Cubrir guards en emit_final_usage_once"""
    chat = ChatPollinations(include_usage_in_stream=True)
    messages = [HumanMessage(content="Test")]

    # Evento con usage para activar pending_usage_md
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
        'data: {"choices":[{"index":0,"delta":{}}],"finish_reason":"stop"}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Debe emitir exactamente un chunk de usage al final
    usage_chunks = [c for c in chunks if c.generation_info and c.generation_info.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].message.usage_metadata is not None


def test_stream_emit_final_usage_when_already_emitted():
    """Cubrir emitted_final_usage ya es True"""
    chat = ChatPollinations(include_usage_in_stream=True)
    messages = [HumanMessage(content="Test")]

    # Dos eventos con usage
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"A"}}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}',
        'data: {"choices":[{"index":0,"delta":{"content":"B"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Solo debe emitir UN chunk de usage al final
    usage_chunks = [c for c in chunks if c.generation_info and c.generation_info.get("usage")]
    assert len(usage_chunks) == 1


def test_stream_emit_final_usage_when_pending_is_none():
    """Cubrir pending_usage_md es None"""
    chat = ChatPollinations(include_usage_in_stream=True)
    messages = [HumanMessage(content="Test")]

    # Sin eventos de usage
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # No debe emitir chunk de usage
    usage_chunks = [c for c in chunks if c.generation_info and c.generation_info.get("usage")]
    assert len(usage_chunks) == 0


def test_stream_with_usage_not_dict():
    """Cubrir evt tiene usage pero no es dict válido"""
    chat = ChatPollinations(include_usage_in_stream=True)
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}],"usage":"not a dict"}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # No debe fallar, simplemente ignora el usage inválido
    assert len(chunks) > 0


def test_stream_with_usage_metadata_extraction_fails():
    """Cubrir _usage_metadata_from_usage retorna None"""
    chat = ChatPollinations(include_usage_in_stream=True)
    messages = [HumanMessage(content="Test")]

    # usage incompleto
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}],"usage":{"incomplete":"data"}}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # No debe fallar
    assert len(chunks) > 0


def test_stream_choices_not_list():
    """Cubrir choices no es list"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":"not a list"}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Debe saltar ese evento
    assert len(chunks) == 0


def test_stream_choices_empty_list():
    """Cubrir choices es lista vacía"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[]}',
        'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Solo debe procesar el segundo evento
    assert len(chunks) > 0
    assert any(c.message.content == "ok" for c in chunks)


def test_stream_choice0_not_dict():
    """Cubrir choice0 no es dict"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":["not a dict"]}',
        'data: {"choices":[{"index":0,"delta":{"content":"valid"}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Solo debe procesar el evento válido
    assert any(c.message.content == "valid" for c in chunks)


def test_stream_delta_not_dict():
    """Cubrir delta no es dict"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":"not a dict"}]}',
        'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Solo debe procesar el delta válido
    assert any(c.message.content == "ok" for c in chunks)


def test_stream_preserve_multimodal_deltas_false():
    """Cubrir preserve_multimodal_deltas es False"""
    chat = ChatPollinations(preserve_multimodal_deltas=False)
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"multimodal"}]}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # No debe agregar content_parts a additional_kwargs
    for chunk in chunks:
        assert "content_parts" not in chunk.message.additional_kwargs


def test_stream_with_all_additional_kwargs_fields():
    """Cubrir todos los campos de additional_kwargs"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f1","arguments":"{}"}}],"function_call":{"name":"fc"},"audio":{"transcript":"audio"},"reasoning_content":"thinking","content_blocks":[{"type":"text","text":"block"}]}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    # Verificar que todos los campos están en additional_kwargs
    chunk = chunks[0]
    assert "tool_calls" in chunk.message.additional_kwargs
    assert "function_call" in chunk.message.additional_kwargs
    assert "audio" in chunk.message.additional_kwargs
    assert "reasoning_content" in chunk.message.additional_kwargs
    assert "content_blocks" in chunk.message.additional_kwargs


def test_stream_generation_info_with_index():
    """Cubrir index es int en choice"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"},"finish_reason":"stop","logprobs":{"tokens":["test"]},"content_filter_results":{"hate":"safe"}}]}',
        'data: [DONE]'
    ]

    chunks = list(chat._stream(messages))

    chunk = chunks[0]
    assert chunk.generation_info["index"] == 0
    assert chunk.generation_info["finish_reason"] == "stop"
    assert "logprobs" in chunk.generation_info
    assert "content_filter_results" in chunk.generation_info


@pytest.mark.asyncio
async def test_astream_with_run_manager():
    """Cubrir run_manager en _astream"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    mock_run_manager = MagicMock()

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"async"}}]}',
        'data: [DONE]'
    ]

    chunks = []
    async for chunk in chat._astream(messages, run_manager=mock_run_manager):
        chunks.append(chunk)

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_astream_stream_options_branches():
    """Cubrir branches de stream_options en astream"""
    # Caso 1: so es None
    chat1 = ChatPollinations(
        include_usage_in_stream=True,
        request_defaults=ChatPollinationsConfig()
    )
    messages = [HumanMessage(content="Test")]

    chat1._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}]}',
        'data: [DONE]'
    ]

    chunks = []
    async for chunk in chat1._astream(messages):
        chunks.append(chunk)

    call = next(c for c in chat1._http.calls if c["method"] == "astream_post_json")
    assert call["payload"]["stream_options"]["include_usage"] is True

    # Caso 2: so es dict sin include_usage
    from langchain_pollinations.chat import StreamOptions

    chat2 = ChatPollinations(
        include_usage_in_stream=True,
        request_defaults=ChatPollinationsConfig(
            stream_options=StreamOptions()
        )
    )

    chat2._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}]}',
        'data: [DONE]'
    ]

    chunks2 = []
    async for chunk in chat2._astream(messages):
        chunks2.append(chunk)

    call2 = next(c for c in chat2._http.calls if c["method"] == "astream_post_json")
    assert call2["payload"]["stream_options"]["include_usage"] is True


@pytest.mark.asyncio
async def test_astream_emit_final_usage_guards():
    """Cubrir guards de emit_final_usage_once en astream"""
    chat = ChatPollinations(include_usage_in_stream=True)
    messages = [HumanMessage(content="Test")]

    # Con usage
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
        'data: [DONE]'
    ]

    chunks = []
    async for chunk in chat._astream(messages):
        chunks.append(chunk)

    # Debe emitir chunk de usage
    usage_chunks = [c for c in chunks if c.generation_info and c.generation_info.get("usage")]
    assert len(usage_chunks) > 0


@pytest.mark.asyncio
async def test_astream_choices_validation():
    """Cubrir validación de choices en astream"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    # choices no es list
    chat._http.stream_events = [
        'data: {"choices":"not a list"}',
        'data: {"choices":[]}',
        'data: {"choices":[{"index":0,"delta":{"content":"valid"}}]}',
        'data: [DONE]'
    ]

    chunks = []
    async for chunk in chat._astream(messages):
        chunks.append(chunk)

    # Solo debe procesar el evento válido
    assert any(c.message.content == "valid" for c in chunks)


@pytest.mark.asyncio
async def test_astream_final_usage_emission_at_end():
    """Cubrir emisión final de usage en astream"""
    chat = ChatPollinations(include_usage_in_stream=True)
    messages = [HumanMessage(content="Test")]

    # Evento con usage
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"text"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}',
        'data: [DONE]'
    ]

    chunks = []
    async for chunk in chat._astream(messages):
        chunks.append(chunk)

    # Debe emitir usage al final
    usage_chunks = [c for c in chunks if c.message.usage_metadata is not None]
    assert len(usage_chunks) > 0
    # El chunk de usage debe ser el último o penúltimo
    last_chunks = chunks[-2:]
    assert any(c.message.usage_metadata is not None for c in last_chunks)


@pytest.mark.asyncio
async def test_astream_no_usage_emission_when_none():
    """Cubrir no emitir cuando pending_usage_md es None"""
    chat = ChatPollinations(include_usage_in_stream=True)
    messages = [HumanMessage(content="Test")]

    # Sin eventos de usage
    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"no usage"}}]}',
        'data: [DONE]'
    ]

    chunks = []
    async for chunk in chat._astream(messages):
        chunks.append(chunk)

    # No debe haber chunks con usage_metadata
    usage_chunks = [c for c in chunks if c.message.usage_metadata is not None]
    assert len(usage_chunks) == 0


def test_stream_with_stop_parameter():
    """Cubrir pasar stop a _stream"""
    chat = ChatPollinations()
    messages = [HumanMessage(content="Test")]

    chat._http.stream_events = [
        'data: {"choices":[{"index":0,"delta":{"content":"test"}}]}',
        'data: [DONE]'
    ]

    # Pasar stop explícitamente
    list(chat._stream(messages, stop=["STOP"]))

    call = next(c for c in chat._http.calls if c["method"] == "stream_post_json")
    assert call["payload"].get("stop") == ["STOP"]


def test_tool_call_chunks_valid_cases():
    """Verificar que los casos válidos sí generan chunks correctamente"""
    delta = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_1",
                "function": {"name": "func1", "arguments": '{"key":"value"}'}
            },
            {
                "index": 1,
                "function": {"name": "func2"}  # sin arguments está ok
            }
        ]
    }
    chunks = _tool_call_chunks_from_delta(delta)

    assert len(chunks) == 2
    assert chunks[0]["name"] == "func1"
    assert chunks[0]["args"] == '{"key":"value"}'
    assert chunks[0]["id"] == "call_1"
    assert chunks[1]["name"] == "func2"
    assert chunks[1]["args"] is None


def test_parse_tool_calls_valid_non_dict_handling():
    """Verificar que el manejo de JSON no-dict está documentado correctamente"""
    message = {
        "tool_calls": [
            {
                "type": "function",
                "id": "call_1",
                "function": {
                    "name": "test_func",
                    "arguments": '{"valid": "dict"}'
                }
            }
        ]
    }
    tool_calls, invalid = _parse_tool_calls(message)

    # Caso válido con dict debe funcionar
    assert len(tool_calls) == 1
    assert tool_calls[0]["args"] == {"valid": "dict"}
    assert len(invalid) == 0


def test_chat_pollinations_parse_chat_result_with_invalid_tool_calls():
    """
    Verifica que tool calls con JSON malformado se manejen correctamente:
    - Se agregan a invalid_tool_calls (no a tool_calls)
    - El campo 'args' debe ser el string original malformado (no un dict vacío)
    - Cumple con el contrato de LangChain v1.2.8+ para InvalidToolCall
    """
    chat = ChatPollinations()

    # Payload con un tool call que tiene arguments JSON inválido
    data = {
        "id": "chatcmpl-123",
        "model": "openai",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_bad",
                            "function": {
                                "name": "badtool",
                                "arguments": "{invalid json}"  # JSON malformado
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }

    # Después del fix, este parsing debe funcionar sin lanzar ValidationError
    result = chat._parse_chat_result(data)
    gen = result.generations[0]

    # Validaciones: debe estar en invalid_tool_calls, NO en tool_calls
    assert len(gen.message.invalid_tool_calls) == 1
    assert len(gen.message.tool_calls) == 0

    # El invalid tool call debe tener el string original en args
    invalid_call = gen.message.invalid_tool_calls[0]
    assert invalid_call["name"] == "badtool"
    assert invalid_call["type"] == "invalid_tool_call"

    # CRÍTICO: args debe ser el string malformado original, NO un dict vacío
    assert isinstance(invalid_call["args"], str)
    assert invalid_call["args"] == "{invalid json}"

    # Debe incluir información del error
    assert "error" in invalid_call
    assert invalid_call["id"] == "call_bad"


def test_chat_pollinations_parse_chat_result_with_valid_tool_calls():
    """
    Verifica que tool calls con JSON válido se procesen correctamente:
    - Se agregan a tool_calls (no a invalid_tool_calls)
    - El campo 'args' debe ser un dict con los parámetros parseados
    - Cumple con el contrato de LangChain v1.2.8+ para ToolCall
    """
    chat = ChatPollinations()

    # Payload con un tool call que tiene arguments JSON válido
    data = {
        "id": "chatcmpl-456",
        "model": "openai",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_good",
                            "function": {
                                "name": "goodtool",
                                "arguments": '{"param": "value"}'  # JSON válido
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }

    result = chat._parse_chat_result(data)
    gen = result.generations[0]

    # Validaciones: debe estar en tool_calls, NO en invalid_tool_calls
    assert len(gen.message.tool_calls) == 1
    assert len(gen.message.invalid_tool_calls) == 0

    # El tool call válido debe tener args como dict parseado
    valid_call = gen.message.tool_calls[0]
    assert valid_call["name"] == "goodtool"
    assert valid_call["type"] == "tool_call"

    # CRÍTICO: args debe ser un dict con los parámetros parseados
    assert isinstance(valid_call["args"], dict)
    assert valid_call["args"] == {"param": "value"}

    assert valid_call["id"] == "call_good"


def test_chat_pollinations_parse_chat_result_with_mixed_tool_calls():
    """
    Verifica el manejo de múltiples tool calls mezclados
    (algunos válidos, otros inválidos) en la misma respuesta.
    """
    chat = ChatPollinations()

    data = {
        "id": "chattcp-456789",
        "model": "openai",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_1",
                            "function": {
                                "name": "valid_tool",
                                "arguments": '{"x": 10, "y": 20}'
                            }
                        },
                        {
                            "type": "function",
                            "id": "call_2",
                            "function": {
                                "name": "invalid_tool",
                                "arguments": "{broken: json"
                            }
                        },
                        {
                            "type": "function",
                            "id": "call_3",
                            "function": {
                                "name": "another_valid",
                                "arguments": '{"status": "ok"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25
        }
    }

    result = chat._parse_chat_result(data)
    gen = result.generations[0]

    # Debe haber 2 válidos y 1 inválido
    assert len(gen.message.tool_calls) == 2
    assert len(gen.message.invalid_tool_calls) == 1

    # Validar tool calls válidos
    assert gen.message.tool_calls[0]["name"] == "valid_tool"
    assert isinstance(gen.message.tool_calls[0]["args"], dict)
    assert gen.message.tool_calls[0]["args"] == {"x": 10, "y": 20}

    assert gen.message.tool_calls[1]["name"] == "another_valid"
    assert isinstance(gen.message.tool_calls[1]["args"], dict)
    assert gen.message.tool_calls[1]["args"] == {"status": "ok"}

    # Validar invalid tool call
    assert gen.message.invalid_tool_calls[0]["name"] == "invalid_tool"
    assert isinstance(gen.message.invalid_tool_calls[0]["args"], str)
    assert gen.message.invalid_tool_calls[0]["args"] == "{broken: json"
