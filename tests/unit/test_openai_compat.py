from __future__ import annotations

import sys
import types
import builtins
import typing
from typing import Any, TypedDict, Callable
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

import langchain_pollinations._openai_compat as _openai_compat


@pytest.fixture()
def _audio_b64() -> str:
    # Base64 "de mentira" para no depender de audio real en tests unitarios.
    return "ZGF0YQ=="


@pytest.fixture()
def _dummy_cache_control() -> dict[str, Any]:
    # Estructura típica de cache_control (se usa en varios normalizadores).
    return {"type": "ephemeral"}


class Test__lc_tool_call_to_openai_tool_call:
    def test_passthrough_openai_compatible_dict(self) -> None:
        tc = {
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
            "id": "call_1",
        }
        out = _openai_compat._lc_tool_call_to_openai_tool_call(tc)
        # Si ya es compatible, debe devolver el mismo dict.
        assert out is tc

    def test_raises_on_non_dict(self) -> None:
        with pytest.raises(TypeError):
            _openai_compat._lc_tool_call_to_openai_tool_call("no-es-dict")  # type: ignore[arg-type]

    def test_converts_name_args_dict_and_preserves_id(self) -> None:
        tc = {"name": "sumar", "args": {"a": 1, "b": 2}, "id": "abc"}
        out = _openai_compat._lc_tool_call_to_openai_tool_call(tc)

        assert out["type"] == "function"
        assert out["function"]["name"] == "sumar"
        assert out["function"]["arguments"] == '{"a": 1, "b": 2}'
        assert out["id"] == "abc"

    def test_args_as_string_is_used_as_arguments(self) -> None:
        tc = {"name": "tool", "args": '{"x": 1}'}
        out = _openai_compat._lc_tool_call_to_openai_tool_call(tc)
        assert out["function"]["arguments"] == '{"x": 1}'

    def test_args_json_dumps_failure_falls_back_to_empty_object(self) -> None:
        # Un set no es serializable por json.dumps.
        tc = {"name": "tool", "args": {"x": {1, 2, 3}}}
        out = _openai_compat._lc_tool_call_to_openai_tool_call(tc)
        assert out["function"]["arguments"] == "{}"


class Test__to_jsonable:
    def test_primitives_and_none(self) -> None:
        assert _openai_compat._to_jsonable(None) is None
        assert _openai_compat._to_jsonable("x") == "x"
        assert _openai_compat._to_jsonable(1) == 1
        assert _openai_compat._to_jsonable(1.5) == 1.5
        assert _openai_compat._to_jsonable(True) is True

    def test_dict_and_list_recursive(self) -> None:
        obj = {"a": 1, 2: {"b": [1, 2, {"c": 3}]}}
        out = _openai_compat._to_jsonable(obj)
        # Las claves deben volverse str.
        assert out["a"] == 1
        assert out["2"]["b"][2]["c"] == 3

    def test_model_dump_is_preferred_when_available(self) -> None:
        class HasModelDump:
            def model_dump(self) -> dict[str, Any]:
                return {"x": 1, "y": {"z": 2}}

        out = _openai_compat._to_jsonable(HasModelDump())
        assert out == {"x": 1, "y": {"z": 2}}

    def test_dict_method_used_when_model_dump_fails(self) -> None:
        class HasBoth:
            def model_dump(self) -> dict[str, Any]:
                raise RuntimeError("boom")

            def dict(self) -> dict[str, Any]:
                return {"y": 2}

        out = _openai_compat._to_jsonable(HasBoth())
        assert out == {"y": 2}

    def test_vars_fallback(self) -> None:
        class Plain:
            def __init__(self) -> None:
                self.a = 1
                self.b = {"c": 2}

        out = _openai_compat._to_jsonable(Plain())
        assert out == {"a": 1, "b": {"c": 2}}


class Test__infer_audio_format_from_mime:
    @pytest.mark.parametrize(
        "mime,expected",
        [
            ("audio/mpeg", "mp3"),
            ("audio/mp3", "mp3"),
            ("audio/wav", "wav"),
            ("audio/wave", "wav"),
            ("audio/x-wav", "wav"),
            ("audio/flac", "flac"),
            ("audio/opus", "opus"),
            ("audio/pcm", "pcm16"),
            ("audio/l16", "pcm16"),
            ("audio/pcm16", "pcm16"),  # fallback por sufijo permitido
        ],
    )
    def test_known_mimes(self, mime: str, expected: str) -> None:
        assert _openai_compat._infer_audio_format_from_mime(mime) == expected

    def test_empty_or_unknown_returns_none(self) -> None:
        assert _openai_compat._infer_audio_format_from_mime("") is None
        assert _openai_compat._infer_audio_format_from_mime("application/json") is None

    def test_whitespace_and_case_is_normalized(self) -> None:
        assert _openai_compat._infer_audio_format_from_mime("  AuDiO/MpEg  ") == "mp3"


class Test__normalize_input_audio_part:
    def test_already_well_formed_is_kept_and_cleaned(self, _audio_b64: str) -> None:
        part = {
            "type": "input_audio",
            "input_audio": {"data": _audio_b64, "format": "mp3"},
            # Campos extra que deben ser eliminados.
            "base64": "x",
            "mime_type": "audio/mpeg",
            "id": "should-go-away",
            "audio": {"data": "x"},
            "data": "x",
            "format": "wav",
        }
        out = _openai_compat._normalize_input_audio_part(part)

        assert out["type"] == "input_audio"
        assert out["input_audio"] == {"data": _audio_b64, "format": "mp3"}
        # Verifica limpieza.
        for k in ("base64", "mime_type", "id", "audio", "data", "format"):
            assert k not in out

    def test_extracts_from_base64_and_mime_type(self, _audio_b64: str) -> None:
        part = {"type": "input_audio", "base64": _audio_b64, "mime_type": "audio/wav"}
        out = _openai_compat._normalize_input_audio_part(part)
        assert out["type"] == "input_audio"
        assert out["input_audio"]["data"] == _audio_b64
        assert out["input_audio"]["format"] == "wav"

    def test_extracts_from_nested_audio_object(self, _audio_b64: str) -> None:
        part = {"type": "input_audio", "audio": {"base64": _audio_b64, "mime_type": "audio/flac"}}
        out = _openai_compat._normalize_input_audio_part(part)
        assert out["input_audio"] == {"data": _audio_b64, "format": "flac"}

    def test_missing_data_defaults_to_empty_string_and_format_defaults_to_mp3(self) -> None:
        part = {"type": "input_audio", "mime_type": "application/octet-stream"}
        out = _openai_compat._normalize_input_audio_part(part)
        assert out["input_audio"]["data"] == ""
        assert out["input_audio"]["format"] == "mp3"

    def test_invalid_format_defaults_to_mp3(self, _audio_b64: str) -> None:
        part = {"type": "input_audio", "data": _audio_b64, "format": "aac"}
        out = _openai_compat._normalize_input_audio_part(part)
        assert out["input_audio"]["format"] == "mp3"


class Test__normalize_video_url_part:
    def test_canonical_nested_form(self) -> None:
        part = {"type": "video_url", "video_url": {"url": "https://x/v.mp4", "mime_type": "video/mp4"}}
        out = _openai_compat._normalize_video_url_part(part)
        assert out == {"type": "video_url", "video_url": {"url": "https://x/v.mp4", "mime_type": "video/mp4"}}

    def test_flat_variant_is_normalized(self) -> None:
        part = {"type": "video_url", "url": "https://x/v.mp4", "mime_type": "video/mp4"}
        out = _openai_compat._normalize_video_url_part(part)
        assert out == {"type": "video_url", "video_url": {"url": "https://x/v.mp4", "mime_type": "video/mp4"}}

    def test_missing_url_returns_type_corrected_but_unmodified(self) -> None:
        part = {"type": "video_url", "mime_type": "video/mp4", "video_url": {"mime_type": "video/mp4"}}
        out = _openai_compat._normalize_video_url_part(part)
        assert out.get("type") == "video_url"
        assert "video_url" in out  # se preserva lo que venga, el backend dará el error final


class Test__normalize_file_part:
    def test_canonical_nested_form_only_non_empty_strings(self) -> None:
        part = {
            "type": "file",
            "file": {
                "file_url": "https://x/doc.pdf",
                "mime_type": "application/pdf",
                "file_name": "",
                "file_id": None,
            },
        }
        out = _openai_compat._normalize_file_part(part)
        assert out == {
            "type": "file",
            "file": {"file_url": "https://x/doc.pdf", "mime_type": "application/pdf"},
        }

    def test_flat_variant_is_normalized(self, _dummy_cache_control: dict[str, Any]) -> None:
        part = {
            "type": "file",
            "file_url": "https://x/doc.pdf",
            "mime_type": "application/pdf",
            "cache_control": _dummy_cache_control,
        }
        out = _openai_compat._normalize_file_part(part)
        assert out["type"] == "file"
        assert out["file"]["file_url"] == "https://x/doc.pdf"
        assert out["file"]["mime_type"] == "application/pdf"
        assert out["cache_control"] == _dummy_cache_control


class Test__normalize_content_part_and__normalize_message_content:
    def test_non_dict_part_is_returned_as_is(self) -> None:
        assert _openai_compat._normalize_content_part("hola") == "hola"

    def test_removes_id_for_unknown_part_type(self) -> None:
        part = {"type": "image_url", "id": "no-permitido", "image_url": {"url": "https://x/img.png"}}
        out = _openai_compat._normalize_content_part(part)
        assert out["type"] == "image_url"
        assert "id" not in out

    def test_text_part_is_strict_and_preserves_cache_control_variants(self, _dummy_cache_control: dict[str, Any]) -> None:
        part = {
            "type": "text",
            "text": "hola",
            "id": "remove",
            "cache_control": _dummy_cache_control,
            "cacheControl": {"type": "other"},
            "cachecontrol": {"type": "another"},
            "extra": "ignored",
        }
        out = _openai_compat._normalize_content_part(part)
        assert out["type"] == "text"
        assert out["text"] == "hola"
        assert out["cache_control"] == _dummy_cache_control
        assert out["cacheControl"] == {"type": "other"}
        assert out["cachecontrol"] == {"type": "another"}
        assert "extra" not in out
        assert "id" not in out

    def test_audio_type_is_converted_to_input_audio(self, _audio_b64: str) -> None:
        part = {"type": "audio", "audio": {"data": _audio_b64, "format": "mp3"}, "id": "remove"}
        out = _openai_compat._normalize_content_part(part)
        assert out["type"] == "input_audio"
        assert out["input_audio"] == {"data": _audio_b64, "format": "mp3"}
        assert "id" not in out

    def test_input_audio_is_normalized(self, _audio_b64: str) -> None:
        part = {"type": "input_audio", "data": _audio_b64, "format": "wav"}
        out = _openai_compat._normalize_content_part(part)
        assert out["type"] == "input_audio"
        assert out["input_audio"]["format"] == "wav"

    def test_video_url_is_normalized(self) -> None:
        part = {"type": "video_url", "url": "https://x/v.mp4"}
        out = _openai_compat._normalize_content_part(part)
        assert out == {"type": "video_url", "video_url": {"url": "https://x/v.mp4"}}

    def test_file_is_normalized(self) -> None:
        part = {"type": "file", "file_url": "https://x/doc.pdf"}
        out = _openai_compat._normalize_content_part(part)
        assert out == {"type": "file", "file": {"file_url": "https://x/doc.pdf"}}

    def test__normalize_message_content_string_and_list_and_other(self) -> None:
        assert _openai_compat._normalize_message_content("hola") == "hola"

        content_list = [
            {"type": "text", "text": "a"},
            {"type": "video_url", "url": "https://x/v.mp4"},
        ]
        out_list = _openai_compat._normalize_message_content(content_list)
        assert isinstance(out_list, list)
        assert out_list[0] == {"type": "text", "text": "a"}
        assert out_list[1] == {"type": "video_url", "video_url": {"url": "https://x/v.mp4"}}

        class Obj:
            def __init__(self) -> None:
                self.x = 1

        out_other = _openai_compat._normalize_message_content(Obj())
        assert out_other == {"x": 1}


class Test__extract_text_from_parts:
    def test_string_returns_itself(self) -> None:
        assert _openai_compat._extract_text_from_parts("hola") == "hola"

    def test_non_list_is_stringified(self) -> None:
        assert _openai_compat._extract_text_from_parts(123) == "123"

    def test_extracts_text_and_content_fields(self) -> None:
        parts = [
            {"type": "text", "text": "uno"},
            {"type": "whatever", "content": "dos"},
            "tres",
            {"type": "text", "text": ""},
        ]
        out = _openai_compat._extract_text_from_parts(parts)
        assert out == "uno\ndos\ntres"

    def test_exclude_thinking_and_redacted_thinking(self) -> None:
        parts = [
            {"type": "thinking", "thinking": "no"},
            {"type": "redacted_thinking", "data": "no"},
            {"type": "text", "text": "sí"},
        ]
        out = _openai_compat._extract_text_from_parts(parts, exclude_thinking=True)
        assert out == "sí"


class Test_lc_messages_to_openai:
    def test_roles_and_tool_message_strictness(self) -> None:
        msgs = [
            SystemMessage(content="sistema"),
            HumanMessage(content="usuario", name="bob"),
            ToolMessage(content=[{"type": "text", "text": "resultado"}], tool_call_id="call_1"),
        ]
        out = _openai_compat.lc_messages_to_openai(msgs)

        assert out[0]["role"] == "system"
        assert out[0]["content"] == "sistema"

        assert out[1]["role"] == "user"
        assert out[1]["content"] == "usuario"
        assert out[1]["name"] == "bob"

        # ToolMessage: content debe ser string y no debe incluir "name".
        assert out[2]["role"] == "tool"
        assert out[2]["content"] == "resultado"
        assert out[2]["tool_call_id"] == "call_1"
        assert "name" not in out[2]

    def test_ai_message_tool_calls_and_additional_kwargs(self) -> None:
        ai = AIMessage(
            content="ok",
            tool_calls=[{"name": "t1", "args": {"x": 1}, "id": "id1"}],
            invalid_tool_calls=[{"name": "t2", "args": '{"y": 2}'}],
            additional_kwargs={
                "function_call": {"name": "legacy", "arguments": {"a": 1}},
                "audio": {"transcript": "hola", "data": "xxx"},
                "cache_control": {"type": "ephemeral"},
            },
        )
        out = _openai_compat.lc_messages_to_openai([ai])[0]

        assert out["role"] == "assistant"
        assert out["content"] == "ok"
        assert out["cache_control"] == {"type": "ephemeral"}

        assert "tool_calls" in out
        assert len(out["tool_calls"]) == 2
        assert out["tool_calls"][0]["type"] == "function"
        assert out["tool_calls"][0]["function"]["name"] == "t1"
        assert out["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
        assert out["tool_calls"][0]["id"] == "id1"

        assert out["tool_calls"][1]["function"]["name"] == "t2"
        assert out["tool_calls"][1]["function"]["arguments"] == '{"y": 2}'

        assert out["function_call"] == {"name": "legacy", "arguments": {"a": 1}}
        assert out["audio"] == {"transcript": "hola", "data": "xxx"}

    def test_ai_message_uses_raw_tool_calls_when_no_tool_calls_lists(self) -> None:
        ai = AIMessage(
            content="ok",
            additional_kwargs={"tool_calls": [{"type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        )
        out = _openai_compat.lc_messages_to_openai([ai])[0]
        assert out["tool_calls"][0]["function"]["name"] == "x"

    def test_fallback_role_from_type_attribute_for_unknown_message_object(self) -> None:
        class DummyMsg:
            # No es instancia de SystemMessage/HumanMessage/ToolMessage/AIMessage.
            type = "developer"

            def __init__(self) -> None:
                self.content = "hola"
                self.additional_kwargs = {}

        out = _openai_compat.lc_messages_to_openai([DummyMsg()])  # type: ignore[arg-type]
        assert out[0]["role"] == "developer"
        assert out[0]["content"] == "hola"


class Test_tool_to_openai_tool:
    @pytest.fixture()
    def _restore_function_calling_module(self) -> Any:
        # Fixture defensiva: guarda/restaura sys.modules para evitar fugas entre tests.
        key = "langchain_core.utils.function_calling"
        prev = sys.modules.get(key, None)
        yield
        if prev is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = prev

    def test_dict_passthrough(self) -> None:
        tool = {"type": "function", "function": {"name": "x", "parameters": {"type": "object", "properties": {}}}}
        out = _openai_compat.tool_to_openai_tool(tool)
        assert out is tool

    def test_uses_langchain_converter_when_schema_is_not_empty(self, _restore_function_calling_module: Any) -> None:
        key = "langchain_core.utils.function_calling"
        mod = types.ModuleType(key)

        def convert_to_openai_tool(_: Any) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {"name": "t", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}},
            }

        mod.convert_to_openai_tool = convert_to_openai_tool  # type: ignore[attr-defined]
        sys.modules[key] = mod

        class Dummy:
            pass

        out = _openai_compat.tool_to_openai_tool(Dummy())
        assert out["type"] == "function"
        assert out["function"]["name"] == "t"
        assert "a" in out["function"]["parameters"]["properties"]

    def test_converter_returning_flat_shape_is_wrapped(self, _restore_function_calling_module: Any) -> None:
        key = "langchain_core.utils.function_calling"
        mod = types.ModuleType(key)

        def convert_to_openai_tool(_: Any) -> dict[str, Any]:
            return {
                "name": "t_flat",
                "description": "desc",
                "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
            }

        mod.convert_to_openai_tool = convert_to_openai_tool  # type: ignore[attr-defined]
        sys.modules[key] = mod

        class Dummy:
            pass

        out = _openai_compat.tool_to_openai_tool(Dummy())
        assert out["type"] == "function"
        assert out["function"]["name"] == "t_flat"
        assert out["function"]["description"] == "desc"

    def test_fallback_when_converter_import_fails(self, _restore_function_calling_module: Any) -> None:
        # Forzamos el fallo del "from ... import convert_to_openai_tool"
        # creando un módulo sin ese símbolo.
        key = "langchain_core.utils.function_calling"
        sys.modules[key] = types.ModuleType(key)

        class ArgsSchema(BaseModel):
            a: str = Field(default="x")

        class DummyTool:
            name = "dummy"
            description = "una herramienta"
            args_schema = ArgsSchema

        out = _openai_compat.tool_to_openai_tool(DummyTool())
        assert out["type"] == "function"
        assert out["function"]["name"] == "dummy"
        assert out["function"]["description"] == "una herramienta"
        assert "properties" in out["function"]["parameters"]
        assert "a" in out["function"]["parameters"]["properties"]

    def test_fallback_uses_tool_call_schema_when_present(self, _restore_function_calling_module: Any) -> None:
        key = "langchain_core.utils.function_calling"
        sys.modules[key] = types.ModuleType(key)

        class ToolCallSchema(BaseModel):
            b: int

        class DummyTool:
            name = "dummy"
            tool_call_schema = ToolCallSchema

        out = _openai_compat.tool_to_openai_tool(DummyTool())
        assert "b" in out["function"]["parameters"]["properties"]

    def test_fallback_uses_get_input_schema_when_present(self, _restore_function_calling_module: Any) -> None:
        key = "langchain_core.utils.function_calling"
        sys.modules[key] = types.ModuleType(key)

        class InputSchema(BaseModel):
            c: str

        class DummyTool:
            name = "dummy"

            def get_input_schema(self) -> type[BaseModel]:
                return InputSchema

        out = _openai_compat.tool_to_openai_tool(DummyTool())
        assert "c" in out["function"]["parameters"]["properties"]

    def test_pydantic_model_class_tool_is_supported(self, _restore_function_calling_module: Any) -> None:
        key = "langchain_core.utils.function_calling"
        sys.modules[key] = types.ModuleType(key)

        class MyToolSchema(BaseModel):
            """Doc de MyToolSchema."""
            d: float

        out = _openai_compat.tool_to_openai_tool(MyToolSchema)
        assert out["function"]["name"] == "MyToolSchema"
        assert "d" in out["function"]["parameters"]["properties"]

    def test_typed_dict_via_type_adapter_is_supported(self, _restore_function_calling_module: Any) -> None:
        key = "langchain_core.utils.function_calling"
        sys.modules[key] = types.ModuleType(key)

        class MyTyped(TypedDict):
            e: int

        out = _openai_compat.tool_to_openai_tool(MyTyped)
        assert out["function"]["name"] == "MyTyped"
        assert "e" in out["function"]["parameters"]["properties"]


class Test__safe_json_loads:
    def test_valid_json(self) -> None:
        assert _openai_compat._safe_json_loads('{"a": 1}') == {"a": 1}

    def test_invalid_json_returns_original_string(self) -> None:
        s = "{not-json"
        assert _openai_compat._safe_json_loads(s) == s


class Test_tool_calls:
    def test_lc_tool_call_arguments_json_serialization_failure(self) -> None:
        """
        Verifica que si 'args' no es serializable, se use '{}' como fallback.
        Cubre la excepción en json.dumps dentro de _lc_tool_call_to_openai_tool_call.
        """
        # Un objeto que falla al ser serializado (set no es JSON serializable)
        unserializable_args = {"param": {1, 2, 3}}

        tc = {
            "name": "test_tool",
            "args": unserializable_args,
            "id": "call_123"
        }

        out = _openai_compat._lc_tool_call_to_openai_tool_call(tc)

        assert out["type"] == "function"
        assert out["function"]["name"] == "test_tool"
        assert out["function"]["arguments"] == "{}"  # Fallback activado


class Test_to_jsonable:
    def test_to_jsonable_model_dump_raises_exception(self) -> None:
        """
        Cubre el bloque except cuando model_dump() falla.
        """
        class BrokenModelDump:
            def model_dump(self):
                raise ValueError("Simulated model_dump failure")

            def dict(self):
                return {"fallback": "dict_method"}

        obj = BrokenModelDump()
        result = _openai_compat._to_jsonable(obj)
        assert result == {"fallback": "dict_method"}

    def test_to_jsonable_dict_raises_exception(self) -> None:
        """
        Cubre el bloque except cuando dict() falla (y no hay model_dump).
        """
        class BrokenDictMethod:
            def dict(self):
                raise ValueError("Simulated dict failure")
            # No tiene model_dump, ni vars (si es builtin o slots sin dict)

        obj = BrokenDictMethod()
        # vars(obj) funciona si tiene __dict__, probemos un objeto simple primero
        # Si vars() funciona, retornará vars.
        obj.x = 1
        result = _openai_compat._to_jsonable(obj)
        assert result == {"x": 1}

    def test_to_jsonable_vars_raises_exception(self) -> None:
        """
        Cubre el bloque except final cuando vars() falla.
        Esto suele pasar con objetos nativos o con __slots__ sin __dict__.
        """
        class SlotsOnly:
            __slots__ = ['x']
            def __init__(self):
                self.x = 1

            # No tiene model_dump, ni dict()
            # vars() fallará con TypeError porque no tiene __dict__

        obj = SlotsOnly()
        # Al fallar todo, debe devolver el objeto original
        result = _openai_compat._to_jsonable(obj)
        assert result is obj


class Test_infer_audio_format_from_mime:
    def test_infer_audio_format_fallback_suffix(self) -> None:
        """
        Cubre la lógica de fallback: split('/') y ver si el sufijo está en _ALLOWED_AUDIO_FORMATS.
        """
        # Caso 1: Sufijo permitido (ej: custom/mp3)
        assert _openai_compat._infer_audio_format_from_mime("application/mp3") == "mp3"

        # Caso 2: Sufijo NO permitido
        assert _openai_compat._infer_audio_format_from_mime("audio/unknown-format") is None

        # Caso 3: Sin barra
        assert _openai_compat._infer_audio_format_from_mime("mp3") is None


class Test__lc_messages_to_openai:
    def test_lc_messages_to_openai_invalid_tool_calls_and_function_call(self) -> None:
        """
        Cubre el bucle de invalid_tool_calls y el manejo de function_call (legacy).
        """
        ai_msg = AIMessage(
            content="",
            # Lista válida vacía, pero invalid_tool_calls presente
            invalid_tool_calls=[
                {"name": "bad_tool", "args": "not-json", "id": "invalid_1", "error": "err"}
            ],
            additional_kwargs={
                "function_call": {"name": "legacy_fn", "arguments": "{}"}
            }
        )

        out_list = _openai_compat.lc_messages_to_openai([ai_msg])
        msg = out_list[0]

        # Verificar invalid_tool_calls procesados
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "bad_tool"

        # Verificar function_call
        assert "function_call" in msg
        assert msg["function_call"]["name"] == "legacy_fn"


class Test__tool_to_openai_tool:
    def test_tool_to_openai_tool_conversion_failures(self) -> None:
        """
        Fuerza fallos en cada estrategia de conversión para cubrir los bloques 'except: pass'.
        Usamos mocks para simular que cada atributo existe pero falla al llamarse.
        """

        # Creamos un objeto que parece tener todos los métodos, pero todos fallan
        class FailTool:
            name = "fail_tool"
            description = "tool that fails everywhere"

            # 1. args_schema falla
            class args_schema:
                @staticmethod
                def model_json_schema():
                    raise ValueError("args_schema failed")

            # 2. tool_call_schema falla
            class tool_call_schema:
                @staticmethod
                def model_json_schema():
                    raise ValueError("tool_call_schema failed")

            # 3. get_input_schema falla
            def get_input_schema(self):
                raise ValueError("get_input_schema failed")

        # Mockeamos convert_to_openai_tool de langchain para que falle también
        with mock.patch("langchain_core.utils.function_calling.convert_to_openai_tool", side_effect=ValueError("convert failed")):
            # Mockeamos pydantic.BaseModel y TypeAdapter para que no interfieran o fallen
            # (Aunque en el código real, si no es subclase, simplemente pasa al siguiente)

            # Ejecutar la conversión
            result = _openai_compat.tool_to_openai_tool(FailTool())

            # Debe haber pasado por todos los excepts y llegado al fallback final
            # Fallback final: name, description, parameters={}
            assert result["type"] == "function"
            assert result["function"]["name"] == "fail_tool"
            assert result["function"]["parameters"] == {
                "type": "object",
                "properties": {},
                "additionalProperties": True
            }

    def test_tool_to_openai_tool_get_input_schema_returns_bad_object(self) -> None:
        """
        Cubre el caso donde get_input_schema() retorna algo, pero ese algo falla en model_json_schema().
        """
        class BadSchemaObj:
            def model_json_schema(self):
                raise ValueError("Boom")

        class ToolWithBadSchema:
            name = "tool_bad_schema"
            def get_input_schema(self):
                return BadSchemaObj()

        with mock.patch("langchain_core.utils.function_calling.convert_to_openai_tool", side_effect=ValueError):
            result = _openai_compat.tool_to_openai_tool(ToolWithBadSchema())

        assert result["function"]["name"] == "tool_bad_schema"
        assert result["function"]["parameters"]["properties"] == {}

    def test_tool_to_openai_tool_pydantic_v1_v2_compat(self) -> None:
        """
        Si model_json_schema falla, el código aún puede inferir schema vía TypeAdapter (Pydantic v2).
        Por eso NO debemos esperar properties vacío.
        """
        from unittest import mock
        from pydantic import BaseModel

        class BrokenPydanticTool(BaseModel):
            x: int

            @classmethod
            def model_json_schema(cls, *args, **kwargs):
                raise ValueError("Pydantic schema gen failed")

        with mock.patch(
            "langchain_core.utils.function_calling.convert_to_openai_tool",
            side_effect=ValueError,
        ):
            result = _openai_compat.tool_to_openai_tool(BrokenPydanticTool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "BrokenPydanticTool"
        # Puede contener 'x' porque entra por TypeAdapter(...).json_schema()
        assert "x" in result["function"]["parameters"].get("properties", {})

    def test_tool_to_openai_tool_type_adapter_failure_falls_back_to_default_schema(self) -> None:
        """
        Fuerza el fallo de TypeAdapter(...).json_schema() para cubrir el bloque except y
        verificar que queda el schema default (properties vacío).
        """
        from unittest import mock
        from pydantic import BaseModel

        class BrokenPydanticTool(BaseModel):
            x: int

            @classmethod
            def model_json_schema(cls, *args, **kwargs):
                raise ValueError("Force BaseModel branch to fail")

        class _FailingTypeAdapter:
            def __init__(self, *_a: Any, **_k: Any) -> None:
                pass

            def json_schema(self) -> dict[str, Any]:
                raise ValueError("Force TypeAdapter.json_schema to fail")

        with mock.patch(
            "langchain_core.utils.function_calling.convert_to_openai_tool",
            side_effect=ValueError,
        ):
            with mock.patch("pydantic.TypeAdapter", _FailingTypeAdapter):
                result = _openai_compat.tool_to_openai_tool(BrokenPydanticTool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "BrokenPydanticTool"
        assert result["function"]["parameters"]["properties"] == {}
        assert result["function"]["parameters"]["additionalProperties"] is True


@pytest.fixture()
def _force_langchain_converter_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Evita que tool_to_openai_tool() retorne temprano por el convertidor de LangChain.

    Forzamos a que convert_to_openai_tool devuelva un schema "vacío" para que el flujo
    continúe hasta las secciones 2.4 y 2.5 (imports de pydantic).
    """
    try:
        import langchain_core.utils.function_calling as _fc  # type: ignore
    except Exception:
        # Si no existe el módulo (raro en este proyecto), igual seguimos: el código ya
        # tiene try/except alrededor de ese import.
        return

    def _empty_converter(_: Any) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "dummy",
                "parameters": {"type": "object", "properties": {}},  # <- schema vacío
            },
        }

    monkeypatch.setattr(_fc, "convert_to_openai_tool", _empty_converter, raising=True)


def _make_import_blocker(
    *,
    block_pydantic_basemodel: bool,
    block_pydantic_typeadapter: bool,
) -> Callable[..., Any]:
    """
    Crea un __import__ que falla solo para 'from pydantic import BaseModel/TypeAdapter'.
    """

    _orig_import = builtins.__import__

    def _custom_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> Any:
        if name == "pydantic":
            fl = set(fromlist or ())
            if block_pydantic_basemodel and "BaseModel" in fl:
                raise ImportError("Blocked pydantic.BaseModel import for coverage")
            if block_pydantic_typeadapter and "TypeAdapter" in fl:
                raise ImportError("Blocked pydantic.TypeAdapter import for coverage")

        return _orig_import(name, globals, locals, fromlist, level)

    return _custom_import


class Test_tool_to_openai_tool_import_failures:
    def test_pydantic_import_basemodel_fails_executes_except(self, monkeypatch: pytest.MonkeyPatch, _force_langchain_converter_empty: None) -> None:
        """
        Cubre:
            except Exception:
                BaseModel = None  # type: ignore
        """
        # Usamos un "tool" que no aporte schema por otras vías (sin args_schema, etc.)
        tool = object()

        monkeypatch.setattr(
            builtins,
            "__import__",
            _make_import_blocker(
                block_pydantic_basemodel=True,
                block_pydantic_typeadapter=False,
            ),
            raising=True,
        )

        out = _openai_compat.tool_to_openai_tool(tool)

        # Validación mínima: debe retornar un tool OpenAI-compatible, sin crashear.
        assert out["type"] == "function"
        assert "function" in out
        assert out["function"]["name"] == "Tool"
        assert "parameters" in out["function"]

    def test_pydantic_import_typeadapter_fails_executes_except(self, monkeypatch: pytest.MonkeyPatch, _force_langchain_converter_empty: None) -> None:
        """
        Cubre:
            except Exception:
                TypeAdapter = None  # type: ignore
        """
        tool = object()

        monkeypatch.setattr(
            builtins,
            "__import__",
            _make_import_blocker(
                block_pydantic_basemodel=False,
                block_pydantic_typeadapter=True,
            ),
            raising=True,
        )

        out = _openai_compat.tool_to_openai_tool(tool)

        # Validación mínima: debe construir el wrapper final y no depender de TypeAdapter.
        assert out["type"] == "function"
        assert out["function"]["name"] == "Tool"
        assert out["function"]["parameters"]["type"] == "object"
        # Como no hay inferencia de schema, nos quedamos con el default vacío.
        assert out["function"]["parameters"]["properties"] == {}
        assert out["function"]["parameters"]["additionalProperties"] is True

    def test_pydantic_imports_basemodel_and_typeadapter_both_fail(self, monkeypatch: pytest.MonkeyPatch, _force_langchain_converter_empty: None) -> None:
        """
        Test adicional (útil para robustez): ambos imports fallan y aun así el
        fallback final debe funcionar.
        """
        tool = object()

        monkeypatch.setattr(
            builtins,
            "__import__",
            _make_import_blocker(
                block_pydantic_basemodel=True,
                block_pydantic_typeadapter=True,
            ),
            raising=True,
        )

        out = _openai_compat.tool_to_openai_tool(tool)

        assert out["type"] == "function"
        assert out["function"]["name"] == "Tool"
        assert out["function"]["parameters"]["properties"] == {}


class Test_UserTier:

    def test_is_accessible_at_module_level(self) -> None:
        # UserTier debe estar exportado en el módulo para que chat.py y código
        # externo puedan importarlo directamente.
        assert hasattr(_openai_compat, "UserTier")

    def test_has_all_five_tier_values(self) -> None:
        # Verifica que los cinco valores del enum del API estén presentes.
        valores = typing.get_args(_openai_compat.UserTier)
        assert set(valores) == {"anonymous", "spore", "seed", "flower", "nectar"}

    def test_has_exactly_five_values(self) -> None:
        # No debe haber más ni menos valores que los cinco documentados en el API.
        assert len(typing.get_args(_openai_compat.UserTier)) == 5

    @pytest.mark.parametrize("tier", ["anonymous", "spore", "seed", "flower", "nectar"])
    def test_each_tier_value_is_a_string(self, tier: str) -> None:
        # Cada valor del Literal debe ser un str (nunca int ni None).
        valores = typing.get_args(_openai_compat.UserTier)
        assert tier in valores
        assert isinstance(tier, str)


class Test__ChatCompletionResponseRequired:

    def test_all_five_base_fields_are_required(self) -> None:
        # Los cinco campos que el API siempre devuelve deben marcarse como requeridos.
        required = _openai_compat._ChatCompletionResponseRequired.__required_keys__
        assert {"id", "object", "created", "model", "choices"} <= required

    def test_no_optional_keys_in_base(self) -> None:
        # La clase base usa total=True (por defecto); ningún campo debe ser opcional.
        optional = _openai_compat._ChatCompletionResponseRequired.__optional_keys__
        assert len(optional) == 0

    def test_annotations_include_id_model_choices(self) -> None:
        # Las anotaciones de la clase base deben incluir los tres campos más críticos.
        hints = typing.get_type_hints(_openai_compat._ChatCompletionResponseRequired)
        assert "id" in hints
        assert "model" in hints
        assert "choices" in hints

    def test_choices_annotation_is_list(self) -> None:
        # choices debe anotarse como alguna variante de list, no como str ni dict plano.
        hints = typing.get_type_hints(_openai_compat._ChatCompletionResponseRequired)
        choices_type = hints["choices"]
        assert typing.get_origin(choices_type) is list

    def test_created_annotation_is_int(self) -> None:
        # created es un Unix timestamp entero, no string ni float.
        hints = typing.get_type_hints(_openai_compat._ChatCompletionResponseRequired)
        assert hints["created"] is int


class Test_ChatCompletionResponse:

    def test_is_accessible_at_module_level(self) -> None:
        # ChatCompletionResponse lo importa chat.py; debe ser público en el módulo.
        assert hasattr(_openai_compat, "ChatCompletionResponse")

    def test_inherits_required_fields_from_base(self) -> None:
        # Los campos de _ChatCompletionResponseRequired deben aparecer como requeridos
        # también en el TypedDict derivado.
        required = _openai_compat.ChatCompletionResponse.__required_keys__
        assert {"id", "object", "created", "model", "choices"} <= required

    def test_user_tier_is_optional(self) -> None:
        # user_tier es el primer campo nuevo y debe ser opcional.
        optional = _openai_compat.ChatCompletionResponse.__optional_keys__
        assert "user_tier" in optional

    def test_citations_is_optional(self) -> None:
        # citations es el segundo campo nuevo y debe ser opcional.
        optional = _openai_compat.ChatCompletionResponse.__optional_keys__
        assert "citations" in optional

    def test_both_fields_are_optional_simultaneously(self) -> None:
        # Verificación conjunta: ambos campos nuevos deben ser opcionales.
        optional = _openai_compat.ChatCompletionResponse.__optional_keys__
        assert {"user_tier", "citations"} <= optional

    def test_pre_existing_optional_fields_remain(self) -> None:
        # Los campos opcionales anteriores no deben haberse perdido al agregar los nuevos.
        optional = _openai_compat.ChatCompletionResponse.__optional_keys__
        assert {"system_fingerprint", "usage", "prompt_filter_results"} <= optional

    def test_required_and_optional_keys_are_disjoint(self) -> None:
        # Un campo no puede ser al mismo tiempo requerido y opcional.
        required = _openai_compat.ChatCompletionResponse.__required_keys__
        optional = _openai_compat.ChatCompletionResponse.__optional_keys__
        assert required.isdisjoint(optional)

    def test_minimal_dict_with_only_required_fields_is_valid(self) -> None:
        # Un dict con solo los cinco campos requeridos debe construirse sin errores
        # y ser indexable como ChatCompletionResponse.
        minimal: _openai_compat.ChatCompletionResponse = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "openai",
            "choices": [],
        }
        assert minimal["id"] == "chatcmpl-abc123"
        assert minimal["model"] == "openai"
        assert minimal["choices"] == []

    def test_dict_with_user_tier_set(self) -> None:
        # Un response con user_tier presente debe poder construirse y leerse
        # sin ningún tratamiento especial.
        data: _openai_compat.ChatCompletionResponse = {
            "id": "chatcmpl-tier",
            "object": "chat.completion",
            "created": 1700000001,
            "model": "gemini",
            "choices": [],
            "user_tier": "flower",
        }
        assert data["user_tier"] == "flower"

    def test_dict_with_citations_set(self) -> None:
        # Un response con citations presente debe poder construirse y sus
        # URLs deben ser accesibles directamente.
        data: _openai_compat.ChatCompletionResponse = {
            "id": "chatcmpl-cit",
            "object": "chat.completion",
            "created": 1700000002,
            "model": "gemini-search",
            "choices": [],
            "citations": [
                "https://example.com/source1",
                "https://example.com/source2",
            ],
        }
        assert len(data["citations"]) == 2
        assert data["citations"][0] == "https://example.com/source1"

    def test_dict_with_all_fields_together(self) -> None:
        # El caso completo: un response con user_tier Y citations, como llegaría
        # de un modelo con búsqueda habilitada (ej. gemini-search, perplexity-reasoning).
        data: _openai_compat.ChatCompletionResponse = {
            "id": "chatcmpl-full",
            "object": "chat.completion",
            "created": 1700000003,
            "model": "perplexity-reasoning",
            "choices": [{"message": {"role": "assistant", "content": "respuesta"}}],
            "user_tier": "nectar",
            "citations": ["https://a.com", "https://b.com", "https://c.com"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        assert data["user_tier"] == "nectar"
        assert len(data["citations"]) == 3
        assert data["usage"]["total_tokens"] == 30

    def test_user_tier_annotation_resolves_to_user_tier_alias(self) -> None:
        # get_type_hints resuelve las anotaciones diferidas (from __future__ import annotations)
        # y debe devolver el mismo objeto que UserTier del módulo.
        hints = typing.get_type_hints(_openai_compat.ChatCompletionResponse)
        assert "user_tier" in hints
        assert hints["user_tier"] is _openai_compat.UserTier

    def test_citations_annotation_resolves_to_list_of_str(self) -> None:
        # citations debe anotarse como list[str], no como list[Any] ni como str.
        hints = typing.get_type_hints(_openai_compat.ChatCompletionResponse)
        assert "citations" in hints
        annotation = hints["citations"]
        # Verifica que sea alguna forma de list parametrizado con str.
        assert typing.get_origin(annotation) is list
        assert typing.get_args(annotation) == (str,)

    @pytest.mark.parametrize("tier", ["anonymous", "spore", "seed", "flower", "nectar"])
    def test_each_user_tier_value_is_assignable(self, tier: str) -> None:
        # Cada uno de los cuatro valores válidos de UserTier debe poder asignarse
        # al campo user_tier de un ChatCompletionResponse sin errores en runtime.
        data: _openai_compat.ChatCompletionResponse = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [],
            "user_tier": tier,  # type: ignore[typeddict-item]
        }
        assert data["user_tier"] == tier

    def test_user_tier_not_in_required_keys(self) -> None:
        # Doble comprobación: user_tier no debe estar en los campos requeridos.
        # Si el API no devuelve user_tier, el response sigue siendo válido.
        required = _openai_compat.ChatCompletionResponse.__required_keys__
        assert "user_tier" not in required

    def test_citations_not_in_required_keys(self) -> None:
        # Doble comprobación: citations no debe estar en los campos requeridos.
        # Modelos sin búsqueda no devuelven citations y eso es correcto.
        required = _openai_compat.ChatCompletionResponse.__required_keys__
        assert "citations" not in required

    def test_is_subclass_of_required_base(self) -> None:
        # ChatCompletionResponse debe ser reconocida como TypedDict que hereda
        # de _ChatCompletionResponseRequired.
        bases = _openai_compat.ChatCompletionResponse.__orig_bases__
        assert _openai_compat._ChatCompletionResponseRequired in bases
