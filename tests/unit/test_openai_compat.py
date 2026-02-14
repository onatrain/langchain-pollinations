import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_pollinations._openai_compat import (
    _lc_tool_call_to_openai_tool_call,
    _to_jsonable,
    _infer_audio_format_from_mime,
    _normalize_input_audio_part,
    _normalize_content_part,
    _normalize_message_content,
    _extract_text_from_parts,
    lc_messages_to_openai,
    tool_to_openai_tool,
    _safe_json_loads,
)


@pytest.fixture
def api_key_from_env(monkeypatch) -> str:
    """
    Lee POLLINATIONS_API_KEY desde .env si existe, y lo inyecta en el entorno.
    Si no existe o no define la variable, usa un valor por defecto para tests.
    """
    env_path = Path(".env")
    api_key = "test_api_key_default"

    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "POLLINATIONS_API_KEY":
                api_key = value.strip().strip("'").strip('"')
                break

    monkeypatch.setenv("POLLINATIONS_API_KEY", api_key)
    return api_key


class MockBaseMessage:
    def __init__(self, content: Any, **kwargs):
        self.content = content
        self.additional_kwargs = kwargs.get("additional_kwargs", {})
        self.name = kwargs.get("name")
        self.type = kwargs.get("type", "base")


class MockSystemMessage(MockBaseMessage):
    def __init__(self, content: Any, **kwargs):
        super().__init__(content, **kwargs)
        self.type = "system"


class MockHumanMessage(MockBaseMessage):
    def __init__(self, content: Any, **kwargs):
        super().__init__(content, **kwargs)
        self.type = "human"


class MockAIMessage(MockBaseMessage):
    def __init__(self, content: Any, **kwargs):
        super().__init__(content, **kwargs)
        self.type = "ai"
        self.tool_calls = kwargs.get("tool_calls", [])
        self.invalid_tool_calls = kwargs.get("invalid_tool_calls", [])


class MockToolMessage(MockBaseMessage):
    def __init__(self, content: Any, tool_call_id: str, **kwargs):
        super().__init__(content, **kwargs)
        self.type = "tool"
        self.tool_call_id = tool_call_id


def test_lc_tool_call_to_openai_tool_call_with_openai_format():
    """Test cuando ya viene en formato OpenAI"""
    tc = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "NYC"}'
        },
        "id": "call_123"
    }

    result = _lc_tool_call_to_openai_tool_call(tc)

    assert result == tc
    assert result["type"] == "function"
    assert result["id"] == "call_123"


def test_lc_tool_call_to_openai_tool_call_with_lc_format():
    """Test conversión desde formato LangChain"""
    tc = {
        "name": "get_weather",
        "args": {"location": "NYC", "units": "celsius"},
        "id": "call_456"
    }

    result = _lc_tool_call_to_openai_tool_call(tc)

    assert result["type"] == "function"
    assert result["function"]["name"] == "get_weather"
    assert json.loads(result["function"]["arguments"]) == {"location": "NYC", "units": "celsius"}
    assert result["id"] == "call_456"


def test_lc_tool_call_to_openai_tool_call_with_string_args():
    """Test cuando args ya es un string JSON"""
    tc = {
        "name": "calculate",
        "args": '{"x": 5, "y": 10}',
        "id": "call_789"
    }

    result = _lc_tool_call_to_openai_tool_call(tc)

    assert result["function"]["arguments"] == '{"x": 5, "y": 10}'


def test_lc_tool_call_to_openai_tool_call_without_id():
    """Test cuando no hay id en el tool call"""
    tc = {
        "name": "test_tool",
        "args": {}
    }

    result = _lc_tool_call_to_openai_tool_call(tc)

    assert "id" not in result
    assert result["type"] == "function"
    assert result["function"]["name"] == "test_tool"


def test_lc_tool_call_to_openai_tool_call_with_invalid_type():
    """Test cuando no es un dict válido"""
    with pytest.raises(TypeError, match="ToolCall must be dict"):
        _lc_tool_call_to_openai_tool_call("not a dict")


def test_lc_tool_call_to_openai_tool_call_with_none_args():
    """Test cuando args es None"""
    tc = {
        "name": "no_args_tool",
        "args": None
    }

    result = _lc_tool_call_to_openai_tool_call(tc)

    assert result["function"]["arguments"] == "{}"


def test_to_jsonable_with_primitives():
    """Test conversión de tipos primitivos"""
    assert _to_jsonable(None) is None
    assert _to_jsonable("text") == "text"
    assert _to_jsonable(42) == 42
    assert _to_jsonable(3.14) == 3.14
    assert _to_jsonable(True) is True


def test_to_jsonable_with_dict():
    """Test conversión de diccionario"""
    data = {"key": "value", "nested": {"num": 123}}
    result = _to_jsonable(data)

    assert result == {"key": "value", "nested": {"num": 123}}


def test_to_jsonable_with_list():
    """Test conversión de lista"""
    data = [1, "two", {"three": 3}]
    result = _to_jsonable(data)

    assert result == [1, "two", {"three": 3}]


def test_to_jsonable_with_tuple():
    """Test conversión de tupla a lista"""
    data = (1, 2, 3)
    result = _to_jsonable(data)

    assert result == [1, 2, 3]


def test_to_jsonable_with_model_dump():
    """Test con objeto que tiene model_dump()"""
    obj = MagicMock()
    obj.model_dump.return_value = {"field": "value"}

    result = _to_jsonable(obj)

    assert result == {"field": "value"}
    obj.model_dump.assert_called_once()


def test_to_jsonable_with_dict_method():
    """Test con objeto que tiene dict()"""
    obj = MagicMock()
    obj.model_dump = None
    obj.dict.return_value = {"field": "value"}

    result = _to_jsonable(obj)

    assert result == {"field": "value"}
    obj.dict.assert_called_once()


def test_to_jsonable_with_vars():
    """Test con objeto usando vars()"""
    class SimpleObj:
        def __init__(self):
            self.field = "value"

    obj = SimpleObj()
    result = _to_jsonable(obj)

    assert result == {"field": "value"}


def test_infer_audio_format_mp3():
    """Test inferencia de formato MP3"""
    assert _infer_audio_format_from_mime("audio/mpeg") == "mp3"
    assert _infer_audio_format_from_mime("audio/mp3") == "mp3"
    assert _infer_audio_format_from_mime("AUDIO/MPEG") == "mp3"


def test_infer_audio_format_wav():
    """Test inferencia de formato WAV"""
    assert _infer_audio_format_from_mime("audio/wav") == "wav"
    assert _infer_audio_format_from_mime("audio/wave") == "wav"
    assert _infer_audio_format_from_mime("audio/x-wav") == "wav"


def test_infer_audio_format_flac():
    """Test inferencia de formato FLAC"""
    assert _infer_audio_format_from_mime("audio/flac") == "flac"


def test_infer_audio_format_opus():
    """Test inferencia de formato Opus"""
    assert _infer_audio_format_from_mime("audio/opus") == "opus"


def test_infer_audio_format_pcm16():
    """Test inferencia de formato PCM16"""
    assert _infer_audio_format_from_mime("audio/pcm") == "pcm16"
    assert _infer_audio_format_from_mime("audio/l16") == "pcm16"


def test_infer_audio_format_from_suffix():
    """Test inferencia desde sufijo del MIME type"""
    assert _infer_audio_format_from_mime("audio/opus") == "opus"


def test_infer_audio_format_empty_or_invalid():
    """Test con MIME type vacío o inválido"""
    assert _infer_audio_format_from_mime("") is None
    assert _infer_audio_format_from_mime("   ") is None
    assert _infer_audio_format_from_mime("text/plain") is None
    assert _infer_audio_format_from_mime("invalid") is None


def test_normalize_input_audio_part_already_correct():
    """Test cuando el part ya está bien formado"""
    part = {
        "type": "input_audio",
        "input_audio": {
            "data": "base64data",
            "format": "mp3"
        }
    }

    result = _normalize_input_audio_part(part)

    assert result["type"] == "input_audio"
    assert result["input_audio"]["data"] == "base64data"
    assert result["input_audio"]["format"] == "mp3"


def test_normalize_input_audio_part_with_base64_and_mime():
    """Test con formato legacy usando base64 y mime_type"""
    part = {
        "type": "input_audio",
        "base64": "encodeddata",
        "mime_type": "audio/mp3"
    }

    result = _normalize_input_audio_part(part)

    assert result["type"] == "input_audio"
    assert result["input_audio"]["data"] == "encodeddata"
    assert result["input_audio"]["format"] == "mp3"
    assert "base64" not in result
    assert "mime_type" not in result


def test_normalize_input_audio_part_with_data_and_format_top_level():
    """Test con data y format en el nivel superior"""
    part = {
        "type": "input_audio",
        "data": "audiodata",
        "format": "wav"
    }

    result = _normalize_input_audio_part(part)

    assert result["input_audio"]["data"] == "audiodata"
    assert result["input_audio"]["format"] == "wav"


def test_normalize_input_audio_part_with_audio_object():
    """Test con objeto audio en el nivel superior"""
    part = {
        "type": "input_audio",
        "audio": {
            "data": "audiodata",
            "format": "flac"
        }
    }

    result = _normalize_input_audio_part(part)

    assert result["input_audio"]["data"] == "audiodata"
    assert result["input_audio"]["format"] == "flac"
    assert "audio" not in result


def test_normalize_input_audio_part_default_format():
    """Test que usa mp3 como formato por defecto"""
    part = {
        "type": "input_audio",
        "data": "somedata"
    }

    result = _normalize_input_audio_part(part)

    assert result["input_audio"]["format"] == "mp3"


def test_normalize_input_audio_part_missing_data():
    """Test cuando falta data"""
    part = {
        "type": "input_audio",
        "format": "mp3"
    }

    result = _normalize_input_audio_part(part)

    assert result["input_audio"]["data"] == ""


def test_normalize_content_part_text():
    """Test normalización de part tipo text"""
    part = {
        "type": "text",
        "text": "Hello world",
        "id": "should_be_removed"
    }

    result = _normalize_content_part(part)

    assert result["type"] == "text"
    assert result["text"] == "Hello world"
    assert "id" not in result


def test_normalize_content_part_text_with_cache_control():
    """Test text part con cache_control"""
    part = {
        "type": "text",
        "text": "Cached text",
        "cache_control": {"type": "ephemeral"}
    }

    result = _normalize_content_part(part)

    assert result["cache_control"] == {"type": "ephemeral"}


def test_normalize_content_part_audio_to_input_audio():
    """Test conversión de tipo 'audio' a 'input_audio'"""
    part = {
        "type": "audio",
        "audio": {
            "data": "audiodata",
            "format": "mp3"
        }
    }

    result = _normalize_content_part(part)

    assert result["type"] == "input_audio"
    assert result["input_audio"]["data"] == "audiodata"


def test_normalize_content_part_non_dict():
    """Test con part que no es dict"""
    part = "plain string"
    result = _normalize_content_part(part)

    assert result == "plain string"


def test_normalize_content_part_removes_id():
    """Test que siempre remueve el campo id"""
    part = {
        "type": "image_url",
        "id": "img_123",
        "image_url": {"url": "https://example.com/img.png"}
    }

    result = _normalize_content_part(part)

    assert "id" not in result
    assert result["image_url"]["url"] == "https://example.com/img.png"


def test_normalize_message_content_string():
    """Test con contenido string"""
    content = "Simple text message"
    result = _normalize_message_content(content)

    assert result == "Simple text message"


def test_normalize_message_content_list():
    """Test con contenido lista de parts"""
    content = [
        {"type": "text", "text": "Hello", "id": "1"},
        {"type": "text", "text": "World", "id": "2"}
    ]

    result = _normalize_message_content(content)

    assert len(result) == 2
    assert result[0]["text"] == "Hello"
    assert "id" not in result[0]
    assert result[1]["text"] == "World"
    assert "id" not in result[1]


def test_normalize_message_content_other():
    """Test con contenido de otro tipo"""
    content = {"custom": "object"}
    result = _normalize_message_content(content)

    assert result == {"custom": "object"}


def test_extract_text_from_parts_string():
    """Test extracción desde string"""
    content = "Plain text"
    result = _extract_text_from_parts(content)

    assert result == "Plain text"


def test_extract_text_from_parts_list_with_text_parts():
    """Test extracción desde lista de text parts"""
    content = [
        {"type": "text", "text": "First"},
        {"type": "text", "text": "Second"}
    ]

    result = _extract_text_from_parts(content)

    assert result == "First\nSecond"


def test_extract_text_from_parts_mixed_list():
    """Test con lista mixta de strings y dicts"""
    content = [
        "Plain string",
        {"type": "text", "text": "Dict text"},
        {"content": "Content field"}
    ]

    result = _extract_text_from_parts(content)

    assert "Plain string" in result
    assert "Dict text" in result
    assert "Content field" in result


def test_extract_text_from_parts_non_list():
    """Test con contenido que no es lista"""
    content = {"some": "dict"}
    result = _extract_text_from_parts(content)

    assert "some" in result


def test_extract_text_from_parts_empty_list():
    """Test con lista vacía"""
    content = []
    result = _extract_text_from_parts(content)

    assert result == ""


def test_lc_messages_to_openai_system_message():
    """Test conversión de SystemMessage"""
    with patch('langchain_pollinations._openai_compat.SystemMessage', MockSystemMessage):
        msg = MockSystemMessage("You are a helpful assistant")

        result = lc_messages_to_openai([msg])

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant"


def test_lc_messages_to_openai_human_message():
    """Test conversión de HumanMessage"""
    with patch('langchain_pollinations._openai_compat.HumanMessage', MockHumanMessage):
        msg = MockHumanMessage("Hello!")

        result = lc_messages_to_openai([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"


def test_lc_messages_to_openai_ai_message():
    """Test conversión de AIMessage"""
    with patch('langchain_pollinations._openai_compat.AIMessage', MockAIMessage):
        msg = MockAIMessage("Hi there!")

        result = lc_messages_to_openai([msg])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there!"


def test_lc_messages_to_openai_ai_message_with_tool_calls():
    """Test AIMessage con tool_calls"""
    with patch('langchain_pollinations._openai_compat.AIMessage', MockAIMessage):
        msg = MockAIMessage(
            "Let me check that",
            tool_calls=[{
                "name": "get_weather",
                "args": {"location": "NYC"},
                "id": "call_1"
            }]
        )

        result = lc_messages_to_openai([msg])

        assert len(result) == 1
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_lc_messages_to_openai_tool_message():
    """Test conversión de ToolMessage"""
    with patch('langchain_pollinations._openai_compat.ToolMessage', MockToolMessage):
        msg = MockToolMessage(
            content="Weather is sunny",
            tool_call_id="call_1"
        )

        result = lc_messages_to_openai([msg])

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "Weather is sunny"
        assert result[0]["tool_call_id"] == "call_1"
        assert "name" not in result[0]


def test_lc_messages_to_openai_message_with_name():
    """Test mensaje con campo name"""
    with patch('langchain_pollinations._openai_compat.HumanMessage', MockHumanMessage):
        msg = MockHumanMessage("Hello", name="John")

        result = lc_messages_to_openai([msg])

        assert result[0]["name"] == "John"


def test_lc_messages_to_openai_message_with_cache_control():
    """Test mensaje con cache_control"""
    with patch('langchain_pollinations._openai_compat.SystemMessage', MockSystemMessage):
        msg = MockSystemMessage(
            "System prompt",
            additional_kwargs={"cache_control": {"type": "ephemeral"}}
        )

        result = lc_messages_to_openai([msg])

        assert "cache_control" in result[0]
        assert result[0]["cache_control"]["type"] == "ephemeral"


def test_lc_messages_to_openai_multipart_content():
    """Test con contenido multipart"""
    with patch('langchain_pollinations._openai_compat.HumanMessage', MockHumanMessage):
        msg = MockHumanMessage([
            {"type": "text", "text": "Look at this", "id": "1"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}, "id": "2"}
        ])

        result = lc_messages_to_openai([msg])

        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
        assert "id" not in result[0]["content"][0]


def test_lc_messages_to_openai_multiple_messages():
    """Test con múltiples mensajes"""
    with patch('langchain_pollinations._openai_compat.SystemMessage', MockSystemMessage), \
         patch('langchain_pollinations._openai_compat.HumanMessage', MockHumanMessage), \
         patch('langchain_pollinations._openai_compat.AIMessage', MockAIMessage):

        messages = [
            MockSystemMessage("System"),
            MockHumanMessage("User"),
            MockAIMessage("Assistant")
        ]

        result = lc_messages_to_openai(messages)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"


def test_tool_to_openai_tool_already_dict():
    """Test cuando ya es un dict en formato OpenAI"""
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}}
        }
    }

    result = tool_to_openai_tool(tool)

    assert result == tool


def test_tool_to_openai_tool_with_args_schema():
    """Test con tool que tiene args_schema"""
    tool = MagicMock()
    tool.name = "calculator"
    tool.description = "Perform calculations"

    schema_mock = MagicMock()
    schema_mock.model_json_schema.return_value = {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"}
        }
    }
    tool.args_schema = schema_mock

    result = tool_to_openai_tool(tool)

    assert result["type"] == "function"
    assert result["function"]["name"] == "calculator"
    assert result["function"]["description"] == "Perform calculations"
    assert "x" in result["function"]["parameters"]["properties"]


def test_tool_to_openai_tool_with_tool_call_schema():
    """Test con tool que tiene tool_call_schema"""
    tool = MagicMock()
    tool.name = "search"
    tool.description = "Search the web"
    tool.args_schema = None

    schema_mock = MagicMock()
    schema_mock.model_json_schema.return_value = {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        }
    }
    tool.tool_call_schema = schema_mock

    result = tool_to_openai_tool(tool)

    assert result["function"]["name"] == "search"
    assert "query" in result["function"]["parameters"]["properties"]


def test_tool_to_openai_tool_with_get_input_schema():
    """Test con tool que tiene get_input_schema()"""
    tool = MagicMock()
    tool.name = "analyzer"
    tool.description = None
    tool.args_schema = None
    tool.tool_call_schema = None

    schema_obj = MagicMock()
    schema_obj.model_json_schema.return_value = {
        "type": "object",
        "properties": {
            "data": {"type": "string"}
        }
    }
    tool.get_input_schema.return_value = schema_obj

    result = tool_to_openai_tool(tool)

    assert result["function"]["name"] == "analyzer"
    assert "data" in result["function"]["parameters"]["properties"]


def test_tool_to_openai_tool_pydantic_model():
    """Test con Pydantic BaseModel como tool"""
    try:
        from pydantic import BaseModel, Field

        class WeatherInput(BaseModel):
            location: str = Field(description="City name")
            units: str = "celsius"

        WeatherInput.__name__ = "get_weather"
        WeatherInput.__doc__ = "Get weather information"

        result = tool_to_openai_tool(WeatherInput)

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert "location" in result["function"]["parameters"]["properties"]
    except ImportError:
        pytest.skip("Pydantic not available")


def test_tool_to_openai_tool_fallback_empty_schema():
    """Test fallback cuando no se puede inferir schema"""
    tool = MagicMock()
    tool.name = "simple_tool"
    tool.description = "A simple tool"
    tool.args_schema = None
    tool.tool_call_schema = None
    tool.get_input_schema = MagicMock(side_effect=Exception("No schema"))

    result = tool_to_openai_tool(tool)

    assert result["type"] == "function"
    assert result["function"]["name"] == "simple_tool"
    assert result["function"]["parameters"]["type"] == "object"


def test_tool_to_openai_tool_uses_docstring_as_description():
    """Test que usa __doc__ como description"""
    tool = MagicMock()
    tool.name = "doc_tool"
    tool.description = None
    tool.__doc__ = "This is from docstring"
    tool.args_schema = None

    result = tool_to_openai_tool(tool)

    assert result["function"]["description"] == "This is from docstring"


def test_tool_to_openai_tool_with_langchain_converter():
    """Test usando el convertidor de LangChain si está disponible"""
    with patch('langchain_core.utils.function_calling.convert_to_openai_tool') as mock_convert:
        mock_convert.return_value = {
            "type": "function",
            "function": {
                "name": "lc_tool",
                "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}
            }
        }

        tool = MagicMock()

        result = tool_to_openai_tool(tool)

        assert result["function"]["name"] == "lc_tool"


def test_safe_json_loads_valid_json():
    """Test con JSON válido"""
    json_str = '{"key": "value", "number": 42}'
    result = _safe_json_loads(json_str)

    assert result == {"key": "value", "number": 42}


def test_safe_json_loads_valid_array():
    """Test con array JSON válido"""
    json_str = '[1, 2, 3, "four"]'
    result = _safe_json_loads(json_str)

    assert result == [1, 2, 3, "four"]


def test_safe_json_loads_invalid_json():
    """Test con JSON inválido devuelve el string original"""
    json_str = '{invalid json}'
    result = _safe_json_loads(json_str)

    assert result == '{invalid json}'


def test_safe_json_loads_empty_string():
    """Test con string vacío"""
    result = _safe_json_loads('')

    assert result == ''


def test_safe_json_loads_malformed_json():
    """Test con JSON mal formado"""
    json_str = '{"key": "value", "missing": }'
    result = _safe_json_loads(json_str)

    assert result == json_str


def test_safe_json_loads_partial_json():
    """Test con JSON parcial"""
    json_str = '{"incomplete":'
    result = _safe_json_loads(json_str)

    assert result == json_str


def test_lc_tool_call_to_openai_tool_call_with_serialization_error_in_args():
    """Test cuando json.dumps falla con los args"""
    class NonSerializable:
        pass

    tc = {
        "name": "tool_with_bad_args",
        "args": NonSerializable()
    }

    result = _lc_tool_call_to_openai_tool_call(tc)

    # Debe manejar el error y devolver "{}"
    assert result["function"]["arguments"] == "{}"


def test_lc_tool_call_to_openai_tool_call_without_name():
    """Test cuando el tool call no tiene name"""
    tc = {
        "args": {"x": 1}
    }

    result = _lc_tool_call_to_openai_tool_call(tc)

    assert result["function"]["name"] == ""
    assert result["type"] == "function"


def test_to_jsonable_with_model_dump_exception():
    """Test cuando model_dump lanza excepción"""
    obj = MagicMock()
    obj.model_dump.side_effect = Exception("Failed to dump")
    obj.dict.return_value = {"fallback": "dict"}

    result = _to_jsonable(obj)

    assert result == {"fallback": "dict"}


def test_to_jsonable_with_dict_method_exception():
    """Test cuando dict() lanza excepción"""
    obj = MagicMock()
    delattr(obj, 'model_dump')
    obj.dict.side_effect = Exception("Failed dict")

    # Debe caer a vars()
    result = _to_jsonable(obj)

    # vars() de MagicMock incluirá atributos internos
    assert isinstance(result, (dict, MagicMock))


def test_to_jsonable_with_vars_exception():
    """Test cuando vars() falla, devuelve el objeto tal cual"""
    # Un objeto sin __dict__ causará que vars() falle
    obj = object()

    result = _to_jsonable(obj)

    # Debe devolver el objeto original
    assert result is obj


def test_infer_audio_format_from_mime_with_unknown_suffix():
    """Test con MIME type con sufijo no reconocido"""
    result = _infer_audio_format_from_mime("audio/unknown-format")

    assert result is None


def test_infer_audio_format_from_mime_without_slash():
    """Test con MIME type sin slash"""
    result = _infer_audio_format_from_mime("audiomp3")

    assert result is None


def test_normalize_input_audio_part_with_audio_base64():
    """Test con audio.base64 en lugar de audio.data"""
    part = {
        "type": "input_audio",
        "audio": {
            "base64": "base64encodeddata",
            "format": "opus"
        }
    }

    result = _normalize_input_audio_part(part)

    assert result["input_audio"]["data"] == "base64encodeddata"
    assert result["input_audio"]["format"] == "opus"


def test_normalize_input_audio_part_with_audio_mime_type():
    """Test con audio.mime_type para inferir formato"""
    part = {
        "type": "input_audio",
        "audio": {
            "data": "somedata",
            "mime_type": "audio/wav"
        }
    }

    result = _normalize_input_audio_part(part)

    assert result["input_audio"]["format"] == "wav"


def test_normalize_input_audio_part_with_invalid_format():
    """Test con formato inválido no en _ALLOWED_AUDIO_FORMATS"""
    part = {
        "type": "input_audio",
        "data": "audiodata",
        "format": "invalid_format"
    }

    result = _normalize_input_audio_part(part)

    # Debe usar mp3 como fallback
    assert result["input_audio"]["format"] == "mp3"


def test_normalize_content_part_with_cache_control_variants():
    """Test con variantes de cache_control (cacheControl, cachecontrol)"""
    part1 = {
        "type": "text",
        "text": "Test",
        "cacheControl": {"type": "ephemeral"}
    }

    result1 = _normalize_content_part(part1)
    assert "cacheControl" in result1

    part2 = {
        "type": "text",
        "text": "Test",
        "cachecontrol": {"type": "ephemeral"}
    }

    result2 = _normalize_content_part(part2)
    assert "cachecontrol" in result2


def test_normalize_content_part_audio_without_input_audio():
    """Test tipo audio con key 'audio' pero sin 'input_audio'"""
    part = {
        "type": "audio",
        "audio": {
            "data": "audiodata",
            "format": "flac"
        }
    }

    result = _normalize_content_part(part)

    assert result["type"] == "input_audio"
    assert result["input_audio"]["data"] == "audiodata"


def test_normalize_content_part_without_type():
    """Test part sin campo type"""
    part = {
        "some_field": "value",
        "id": "remove_me"
    }

    result = _normalize_content_part(part)

    # Debe remover id pero mantener el resto
    assert "id" not in result
    assert result["some_field"] == "value"


def test_extract_text_from_parts_with_content_field():
    """Test extracción de campo 'content' cuando no hay 'text'"""
    content = [
        {"type": "something", "content": "Text from content field"}
    ]

    result = _extract_text_from_parts(content)

    assert "Text from content field" in result


def test_extract_text_from_parts_with_empty_strings():
    """Test que filtra strings vacíos"""
    content = [
        {"type": "text", "text": ""},
        {"type": "text", "text": "Valid text"},
        {"type": "text", "text": ""}
    ]

    result = _extract_text_from_parts(content)

    # Solo debe incluir el texto válido
    assert result == "Valid text"


def test_lc_messages_to_openai_ai_message_with_invalid_tool_calls():
    """Test AIMessage con invalid_tool_calls"""
    with patch('langchain_pollinations._openai_compat.AIMessage', MockAIMessage):
        msg = MockAIMessage(
            "Response",
            invalid_tool_calls=[{
                "name": "invalid_tool",
                "args": {"bad": "args"},
                "id": "invalid_1"
            }]
        )

        result = lc_messages_to_openai([msg])

        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1


def test_lc_messages_to_openai_ai_message_with_raw_tool_calls():
    """Test AIMessage con tool_calls en additional_kwargs"""
    with patch('langchain_pollinations._openai_compat.AIMessage', MockAIMessage):
        msg = MockAIMessage(
            "Response",
            additional_kwargs={
                "tool_calls": [
                    {"type": "function", "function": {"name": "raw_tool"}}
                ]
            }
        )

        result = lc_messages_to_openai([msg])

        assert "tool_calls" in result[0]


def test_lc_messages_to_openai_ai_message_with_function_call():
    """Test AIMessage con function_call en additional_kwargs"""
    with patch('langchain_pollinations._openai_compat.AIMessage', MockAIMessage):
        msg = MockAIMessage(
            "Response",
            additional_kwargs={
                "function_call": {"name": "legacy_function", "arguments": "{}"}
            }
        )

        result = lc_messages_to_openai([msg])

        assert "function_call" in result[0]
        assert result[0]["function_call"]["name"] == "legacy_function"


def test_lc_messages_to_openai_ai_message_with_audio():
    """Test AIMessage con audio en additional_kwargs"""
    with patch('langchain_pollinations._openai_compat.AIMessage', MockAIMessage):
        msg = MockAIMessage(
            "Response",
            additional_kwargs={
                "audio": {"id": "audio_123", "transcript": "Hello"}
            }
        )

        result = lc_messages_to_openai([msg])

        assert "audio" in result[0]
        assert result[0]["audio"]["id"] == "audio_123"


def test_lc_messages_to_openai_tool_message_with_multipart_content():
    """Test ToolMessage con contenido multipart que debe convertirse a string"""
    with patch('langchain_pollinations._openai_compat.ToolMessage', MockToolMessage):
        msg = MockToolMessage(
            content=[
                {"type": "text", "text": "Result part 1"},
                {"type": "text", "text": "Result part 2"}
            ],
            tool_call_id="call_123"
        )

        result = lc_messages_to_openai([msg])

        # El contenido debe ser un string
        assert isinstance(result[0]["content"], str)
        assert "Result part 1" in result[0]["content"]


def test_lc_messages_to_openai_message_without_name_attribute():
    """Test mensaje sin atributo name"""
    with patch('langchain_pollinations._openai_compat.HumanMessage', MockHumanMessage):
        msg = MockHumanMessage("Hello")
        delattr(msg, 'name')

        result = lc_messages_to_openai([msg])

        assert "name" not in result[0]


def test_tool_to_openai_tool_with_lc_converter_returning_name_parameters():
    """Test cuando el convertidor LC devuelve formato {name, parameters, description}"""
    with patch('langchain_core.utils.function_calling.convert_to_openai_tool') as mock_convert:
        mock_convert.return_value = {
            "name": "converted_tool",
            "description": "Converted by LC",
            "parameters": {
                "type": "object",
                "properties": {"param": {"type": "string"}}
            }
        }

        tool = MagicMock()

        result = tool_to_openai_tool(tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "converted_tool"
        assert result["function"]["description"] == "Converted by LC"


def test_tool_to_openai_tool_with_lc_converter_empty_schema():
    """Test cuando LC converter devuelve schema vacío, debe continuar con fallback"""
    with patch('langchain_core.utils.function_calling.convert_to_openai_tool') as mock_convert:
        mock_convert.return_value = {
            "type": "function",
            "function": {
                "name": "empty_tool",
                "parameters": {}
            }
        }

        tool = MagicMock()
        tool.name = "fallback_name"
        tool.args_schema = None

        result = tool_to_openai_tool(tool)

        # Debe intentar usar el fallback
        assert result["type"] == "function"


def test_tool_to_openai_tool_with_lc_converter_exception():
    """Test cuando el convertidor LC lanza excepción"""
    with patch('langchain_core.utils.function_calling.convert_to_openai_tool') as mock_convert:
        mock_convert.side_effect = Exception("Conversion failed")

        tool = MagicMock()
        tool.name = "tool_name"
        tool.description = "Tool description"
        tool.args_schema = None

        result = tool_to_openai_tool(tool)

        # Debe usar el fallback
        assert result["type"] == "function"
        assert result["function"]["name"] == "tool_name"


def test_tool_to_openai_tool_args_schema_exception():
    """Test cuando args_schema.model_json_schema() lanza excepción"""
    tool = MagicMock()
    tool.name = "error_tool"
    tool.args_schema = MagicMock()
    tool.args_schema.model_json_schema.side_effect = Exception("Schema error")

    result = tool_to_openai_tool(tool)

    # Debe continuar con otros métodos de inferencia
    assert result["type"] == "function"


def test_tool_to_openai_tool_tool_call_schema_exception():
    """Test cuando tool_call_schema lanza excepción"""
    tool = MagicMock()
    tool.name = "error_tool"
    tool.args_schema = None
    tool.tool_call_schema = MagicMock()
    tool.tool_call_schema.model_json_schema.side_effect = Exception("Schema error")

    result = tool_to_openai_tool(tool)

    assert result["type"] == "function"


def test_tool_to_openai_tool_get_input_schema_exception():
    """Test cuando get_input_schema() lanza excepción"""
    tool = MagicMock()
    tool.name = "error_tool"
    tool.args_schema = None
    tool.tool_call_schema = None
    tool.get_input_schema.side_effect = Exception("Input schema error")

    result = tool_to_openai_tool(tool)

    assert result["type"] == "function"


def test_tool_to_openai_tool_pydantic_not_installed():
    """Test cuando Pydantic no está disponible"""
    with patch.dict('sys.modules', {'pydantic': None}):
        tool = MagicMock()
        tool.name = "no_pydantic_tool"
        tool.args_schema = None

        result = tool_to_openai_tool(tool)

        assert result["type"] == "function"


def test_tool_to_openai_tool_typeadapter_not_available():
    """Test cuando TypeAdapter no está disponible en Pydantic"""
    tool = MagicMock()
    tool.name = "no_adapter_tool"
    tool.args_schema = None
    tool.tool_call_schema = None
    tool.get_input_schema = MagicMock(side_effect=Exception("No schema"))

    with patch('pydantic.TypeAdapter', None):
        result = tool_to_openai_tool(tool)

        assert result["type"] == "function"


def test_tool_to_openai_tool_typeadapter_exception():
    """Test cuando TypeAdapter lanza excepción"""
    tool = MagicMock()
    tool.name = "adapter_error_tool"
    tool.args_schema = None
    tool.tool_call_schema = None
    tool.get_input_schema = MagicMock(side_effect=Exception("No schema"))

    with patch('pydantic.TypeAdapter') as mock_adapter:
        mock_adapter.side_effect = Exception("TypeAdapter failed")

        result = tool_to_openai_tool(tool)

        assert result["type"] == "function"


def test_tool_to_openai_tool_without_description():
    """Test tool sin description ni __doc__"""
    tool = MagicMock()
    tool.name = "no_desc_tool"
    tool.description = None
    tool.__doc__ = None
    tool.args_schema = None

    result = tool_to_openai_tool(tool)

    # No debe tener campo description
    assert "description" not in result["function"]


def test_tool_to_openai_tool_with_empty_docstring():
    """Test tool con __doc__ vacío o solo espacios"""
    tool = MagicMock()
    tool.name = "empty_doc_tool"
    tool.description = None
    tool.__doc__ = "   \n  "
    tool.args_schema = None

    result = tool_to_openai_tool(tool)

    assert "description" not in result["function"]


def test_tool_to_openai_tool_without_name():
    """Test tool sin name ni __name__"""
    tool = MagicMock()
    delattr(tool, 'name')
    delattr(tool, '__name__')
    tool.description = None
    tool.args_schema = None

    result = tool_to_openai_tool(tool)

    # Debe usar "Tool" como nombre por defecto
    assert result["function"]["name"] == "Tool"


def test_tool_to_openai_tool_with_schema_with_oneof():
    """Test schema que tiene oneOf no se considera vacío"""
    tool = MagicMock()
    tool.name = "oneof_tool"

    schema_mock = MagicMock()
    schema_mock.model_json_schema.return_value = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    tool.args_schema = schema_mock

    result = tool_to_openai_tool(tool)

    assert "oneOf" in result["function"]["parameters"]


def test_tool_to_openai_tool_with_schema_with_required():
    """Test schema con required no vacío no se considera vacío"""
    tool = MagicMock()
    tool.name = "required_tool"

    schema_mock = MagicMock()
    schema_mock.model_json_schema.return_value = {
        "type": "object",
        "properties": {},
        "required": ["field1"]
    }
    tool.args_schema = schema_mock

    result = tool_to_openai_tool(tool)

    assert "required" in result["function"]["parameters"]


def test_normalize_input_audio_part_all_fields_cleanup():
    """Test que se eliminan todos los campos extra"""
    part = {
        "type": "input_audio",
        "input_audio": {
            "data": "audiodata",
            "format": "mp3"
        },
        "base64": "should_be_removed",
        "mime_type": "should_be_removed",
        "id": "should_be_removed",
        "audio": "should_be_removed",
        "data": "should_be_removed",
        "format": "should_be_removed"
    }

    result = _normalize_input_audio_part(part)

    assert "base64" not in result
    assert "mime_type" not in result
    assert "id" not in result
    assert "audio" not in result
    # data y format solo deben estar en input_audio
    assert "data" not in result
    assert "format" not in result
    assert result["input_audio"]["data"] == "audiodata"


def test_lc_messages_to_openai_with_empty_name():
    """Test mensaje con name vacío no debe incluirse"""
    with patch('langchain_pollinations._openai_compat.HumanMessage', MockHumanMessage):
        msg = MockHumanMessage("Hello", name="")

        result = lc_messages_to_openai([msg])

        assert "name" not in result[0]


def test_lc_messages_to_openai_ai_message_both_tool_calls_and_raw():
    """Test AIMessage con tool_calls y raw tool_calls en additional_kwargs"""
    with patch('langchain_pollinations._openai_compat.AIMessage', MockAIMessage):
        msg = MockAIMessage(
            "Response",
            tool_calls=[{
                "name": "tool1",
                "args": {},
                "id": "call_1"
            }],
            additional_kwargs={
                "tool_calls": [
                    {"type": "function", "function": {"name": "raw_tool"}}
                ]
            }
        )

        result = lc_messages_to_openai([msg])

        # Debe priorizar los tool_calls del mensaje sobre los de additional_kwargs
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "tool1"
