from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

# Formatos permitidos por api.json para input_audio.format
_ALLOWED_AUDIO_FORMATS = {"wav", "mp3", "flac", "opus", "pcm16"}


def _lc_tool_call_to_openai_tool_call(tc: Any) -> dict[str, Any]:
    """
    Convierte LangChain-like ToolCall dict a OpenAI tool_call dict:
      {"id": "...", "type":"function", "function":{"name":"...", "arguments":"{...json...}"}}
    """
    if isinstance(tc, dict) and tc.get("type") == "function" and isinstance(tc.get("function"), dict):
        return cast(dict[str, Any], tc)

    if not isinstance(tc, dict):
        raise TypeError("ToolCall must be dict or OpenAI compatible dict.")

    name = tc.get("name") or ""
    args = tc.get("args", {})
    if isinstance(args, str):
        arguments = args
    else:
        try:
            arguments = json.dumps(args if args is not None else {})
        except Exception:
            arguments = "{}"

    out: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if isinstance(tc.get("id"), str):
        out["id"] = tc["id"]
    return out


def _to_jsonable(obj: Any) -> Any:
    """Conversión a tipos de Python serializables en JSON."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:
            pass

    dict_meth = getattr(obj, "dict", None)
    if callable(dict_meth):
        try:
            return _to_jsonable(obj.dict())
        except Exception:
            pass

    try:
        return _to_jsonable(vars(obj))
    except Exception:
        return obj


def _infer_audio_format_from_mime(mime_type: str) -> str | None:
    mt = mime_type.strip().lower()
    if not mt:
        return None

    # Casos comunes
    if mt in {"audio/mpeg", "audio/mp3"}:
        return "mp3"
    if mt in {"audio/wav", "audio/wave", "audio/x-wav"}:
        return "wav"
    if mt in {"audio/flac"}:
        return "flac"
    if mt in {"audio/opus"}:
        return "opus"
    if mt in {"audio/pcm", "audio/l16"}:
        # El schema usa pcm16, no pcm
        return "pcm16"

    # fallback: usa el sufijo si coincide con allowed
    if "/" in mt:
        suffix = mt.split("/", 1)[1].strip()
        if suffix in _ALLOWED_AUDIO_FORMATS:
            return suffix

    return None


def _normalize_input_audio_part(part: dict[str, Any]) -> dict[str, Any]:
    """
    Normaliza cualquier variante 'input_audio' para que cumpla:
      {"type":"input_audio", "input_audio":{"data": "...base64...", "format":"mp3"}}

    Acepta variantes comunes:
      - ya-correcta: part["input_audio"] = {"data":..., "format":...}
      - legacy/LC: part["base64"]=..., part["mime_type"]="audio/mp3"
      - alt: part["data"]=..., part["format"]="mp3" (top-level)
      - alt: part["audio"]={"data"/"base64", ...} (top-level)
    """
    # 1) Si ya viene bien formada, sólo aseguramos que sea dict jsonable.
    ia = part.get("input_audio")
    if isinstance(ia, dict):
        data = ia.get("data")
        fmt = ia.get("format")
        if isinstance(data, str) and data and isinstance(fmt, str) and fmt in _ALLOWED_AUDIO_FORMATS:
            out = dict(part)
            out["type"] = "input_audio"
            out["input_audio"] = {"data": data, "format": fmt}
            # Limpieza de posibles campos extra que confunden
            out.pop("base64", None)
            out.pop("mime_type", None)
            out.pop("id", None)
            out.pop("audio", None)
            out.pop("data", None)
            out.pop("format", None)
            return out

    # 2) Extrae data/format desde otras formas.
    data: str | None = None
    fmt: str | None = None

    if isinstance(part.get("base64"), str):
        data = cast(str, part.get("base64"))
    elif isinstance(part.get("data"), str):
        data = cast(str, part.get("data"))

    if isinstance(part.get("format"), str):
        fmt = cast(str, part.get("format"))

    if fmt is None and isinstance(part.get("mime_type"), str):
        fmt = _infer_audio_format_from_mime(cast(str, part.get("mime_type")))

    # 3) Variante: part["audio"]={...}
    audio_obj = part.get("audio")
    if isinstance(audio_obj, dict):
        if data is None and isinstance(audio_obj.get("data"), str):
            data = cast(str, audio_obj.get("data"))
        if data is None and isinstance(audio_obj.get("base64"), str):
            data = cast(str, audio_obj.get("base64"))
        if fmt is None and isinstance(audio_obj.get("format"), str):
            fmt = cast(str, audio_obj.get("format"))
        if fmt is None and isinstance(audio_obj.get("mime_type"), str):
            fmt = _infer_audio_format_from_mime(cast(str, audio_obj.get("mime_type")))

    if not (isinstance(data, str) and data):
        # Deja que el backend dé un error más claro, pero ya con la estructura esperada.
        data = ""

    if not (isinstance(fmt, str) and fmt in _ALLOWED_AUDIO_FORMATS):
        # si no podemos inferir, mp3 es el caso más común (y tu ejemplo es mp3)
        fmt = "mp3"

    out = dict(part)
    out["type"] = "input_audio"
    out["input_audio"] = {"data": data, "format": fmt}

    # Limpieza: evita que el gateway interprete mal el part.
    out.pop("base64", None)
    out.pop("mime_type", None)
    out.pop("id", None)
    out.pop("audio", None)
    out.pop("data", None)
    out.pop("format", None)

    return out


def _normalize_content_part(part: Any) -> Any:
    """
    Normaliza un content part a Pollinations/OpenAI chat.completions.
    """
    part = _to_jsonable(part)
    if not isinstance(part, dict):
        return part

    # El API del proveedor NO permite `id` dentro de MessageContentPart
    if "id" in part:
        part = dict(part)
        part.pop("id", None)

    t = part.get("type")
    if not isinstance(t, str):
        return part

    # Limpieza específica por tipo
    if t == "text":
        # dejar solo lo que el schema espera
        out = {"type": "text", "text": part.get("text", "")}
        if "cache_control" in part:
            out["cache_control"] = part["cache_control"]
        if "cacheControl" in part:
            out["cacheControl"] = part["cacheControl"]
        if "cachecontrol" in part:
            out["cachecontrol"] = part["cachecontrol"]
        return out

    # Compatibilidad con variantes donde el tipo llega como "audio"
    if t == "audio":
        # convierte a input_audio y normaliza cuerpo
        p2 = dict(part)
        p2["type"] = "input_audio"
        # algunos payloads usan key "audio" en vez de input_audio
        if "audio" in p2 and "input_audio" not in p2:
            p2["input_audio"] = p2.get("audio")
        return _normalize_input_audio_part(p2)

    # base64/mime_type sin input_audio
    if t == "input_audio":
        return _normalize_input_audio_part(part)

    # Otros tipos (image_url / video_url / file) se dejan pasar, pero ya sin `id`
    return part


def _normalize_message_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_normalize_content_part(p) for p in content]
    return _to_jsonable(content)


def _extract_text_from_parts(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    chunks: list[str] = []
    for p in content:
        if isinstance(p, str):
            chunks.append(p)
        elif isinstance(p, dict):
            # soporta tanto {type:text,text:...} como variantes
            if p.get("type") == "text" and isinstance(p.get("text"), str):
                chunks.append(p["text"])
            elif isinstance(p.get("content"), str):
                chunks.append(p["content"])
    return "\n".join([c for c in chunks if c])


def lc_messages_to_openai(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """
    Convierte mensajes LangChain a mensajes Pollinations/OpenAI-compatible chat.completions.
    """
    out: list[dict[str, Any]] = []

    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, ToolMessage):
            role = "tool"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            role = getattr(m, "type", "user")

        content = _normalize_message_content(m.content)

        # Si es ToolMessage: el API es más estricto, mandar string y evitar `name`
        if isinstance(m, ToolMessage):
            content = _extract_text_from_parts(content)  # <- string
            msg: dict[str, Any] = {"role": "tool", "content": content, "tool_call_id": m.tool_call_id}
            # NO incluir msg["name"]
        else:
            msg = {"role": role, "content": content}
            name = getattr(m, "name", None)
            if isinstance(name, str) and name:
                msg["name"] = name

        cache_control = m.additional_kwargs.get("cache_control")
        if cache_control is not None:
            msg["cache_control"] = _to_jsonable(cache_control)

        if isinstance(m, ToolMessage):
            msg["tool_call_id"] = m.tool_call_id

        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None) or []
            invalid_tool_calls = getattr(m, "invalid_tool_calls", None) or []

            openai_tool_calls: list[dict[str, Any]] = []
            for tc in tool_calls:
                openai_tool_calls.append(_lc_tool_call_to_openai_tool_call(tc))
            for itc in invalid_tool_calls:
                openai_tool_calls.append(_lc_tool_call_to_openai_tool_call(itc))

            if openai_tool_calls:
                msg["tool_calls"] = openai_tool_calls
            else:
                raw_tool_calls = m.additional_kwargs.get("tool_calls")
                if raw_tool_calls is not None:
                    msg["tool_calls"] = _to_jsonable(raw_tool_calls)

            function_call = m.additional_kwargs.get("function_call")
            if function_call is not None:
                msg["function_call"] = _to_jsonable(function_call)

            audio = m.additional_kwargs.get("audio")
            if audio is not None:
                msg["audio"] = _to_jsonable(audio)

        out.append(msg)

    return out


def tool_to_openai_tool(tool: Any) -> dict[str, Any]:
    """
    Convierte herramientas LangChain (BaseTool/StructuredTool/Pydantic model/TypedDict) al esquema:
    {"type":"function","function":{"name":..., "description":..., "parameters":...}}

    Garantía clave: cuando sea posible, "parameters" NO queda vacío ({}) para structured output.
    """

    def _is_schema_empty(params: Any) -> bool:
        if not isinstance(params, dict):
            return True
        # schema vacío o sin propiedades útiles
        props = params.get("properties")
        if isinstance(props, dict) and len(props) > 0:
            return False
        # a veces TypeAdapter produce $defs y properties vacío; igual lo consideramos vacío para tools
        if any(k in params for k in ("oneOf", "anyOf", "allOf")):
            return False
        # si tiene "required" no vacío, tampoco está vacío
        req = params.get("required")
        if isinstance(req, list) and len(req) > 0:
            return False
        return True

    def _wrap_as_function(name: str, description: str | None, parameters: dict[str, Any]) -> dict[str, Any]:
        fn: dict[str, Any] = {"name": name, "parameters": parameters}
        if isinstance(description, str) and description.strip():
            fn["description"] = description.strip()
        return {"type": "function", "function": fn}

    # 0) Si ya viene dict en formato OpenAI, úsalo tal cual
    if isinstance(tool, dict):
        return cast(dict[str, Any], tool)

    # 1) Mejor ruta: convertidor oficial de LangChain (maneja más casos que nuestras heurísticas)
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool as _lc_convert  # type: ignore
    except Exception:
        _lc_convert = None

    if _lc_convert is not None:
        try:
            converted = _lc_convert(tool)
            if isinstance(converted, dict):
                # Caso A: ya viene como {"type":"function","function":{...}}
                if converted.get("type") == "function" and isinstance(converted.get("function"), dict):
                    fn = converted.get("function") or {}
                    params = fn.get("parameters")
                    # Si el schema quedó vacío, seguimos intentando inferirlo abajo
                    if not _is_schema_empty(params):
                        return cast(dict[str, Any], converted)

                # Caso B: algunas versiones devuelven {"name","description","parameters"}
                if "name" in converted and "parameters" in converted and isinstance(converted.get("name"), str):
                    params = converted.get("parameters")
                    if isinstance(params, dict) and not _is_schema_empty(params):
                        return _wrap_as_function(
                            name=cast(str, converted["name"]),
                            description=cast(str | None, converted.get("description")),
                            parameters=cast(dict[str, Any], params),
                        )
        except Exception:
            pass

    # 2) Fallback robusto: intentar inferir schema de Pydantic/TypedDict/typing
    name = getattr(tool, "name", None) or getattr(tool, "__name__", None) or "Tool"
    description = getattr(tool, "description", None) or (getattr(tool, "__doc__", "") or "").strip() or None

    # default (si no podemos inferir)
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": True}

    # 2.1) args_schema (BaseTool clásico)
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        try:
            mj = getattr(args_schema, "model_json_schema", None)
            if callable(mj):
                candidate = cast(dict[str, Any], args_schema.model_json_schema())
                if not _is_schema_empty(candidate):
                    parameters = candidate
        except Exception:
            pass

    # 2.2) tool_call_schema (StructuredTool / otros)
    if _is_schema_empty(parameters):
        tcs = getattr(tool, "tool_call_schema", None)
        if tcs is not None:
            try:
                mj = getattr(tcs, "model_json_schema", None)
                if callable(mj):
                    candidate = cast(dict[str, Any], tcs.model_json_schema())
                    if not _is_schema_empty(candidate):
                        parameters = candidate
            except Exception:
                pass

    # 2.3) get_input_schema()
    if _is_schema_empty(parameters):
        gis = getattr(tool, "get_input_schema", None)
        if callable(gis):
            try:
                schema_obj = gis()
                mj = getattr(schema_obj, "model_json_schema", None)
                if callable(mj):
                    candidate = cast(dict[str, Any], schema_obj.model_json_schema())
                    if not _is_schema_empty(candidate):
                        parameters = candidate
            except Exception:
                pass

    # 2.4) Pydantic model class (BaseModel)
    if _is_schema_empty(parameters):
        try:
            from pydantic import BaseModel  # type: ignore
        except Exception:
            BaseModel = None  # type: ignore

        try:
            if BaseModel is not None and isinstance(tool, type) and issubclass(tool, BaseModel):  # type: ignore[arg-type]
                candidate = cast(dict[str, Any], tool.model_json_schema())  # type: ignore[attr-defined]
                if not _is_schema_empty(candidate):
                    parameters = candidate
        except Exception:
            pass

    # 2.5) TypedDict / typing types via TypeAdapter (Pydantic v2)
    if _is_schema_empty(parameters):
        try:
            from pydantic import TypeAdapter  # type: ignore
        except Exception:
            TypeAdapter = None  # type: ignore

        if TypeAdapter is not None:
            try:
                candidate = cast(dict[str, Any], TypeAdapter(tool).json_schema())  # type: ignore[misc]
                if not _is_schema_empty(candidate):
                    parameters = candidate
            except Exception:
                pass

    return _wrap_as_function(name=str(name), description=cast(str | None, description), parameters=parameters)


def safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return s
