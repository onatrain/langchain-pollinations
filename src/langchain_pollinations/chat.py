from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, Annotated, Literal, Optional, Union, cast

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import ToolCallChunk, tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter

from langchain_pollinations._auth import AuthConfig
from langchain_pollinations._client import HttpConfig, PollinationsHttpClient
from langchain_pollinations._openai_compat import lc_messages_to_openai, tool_to_openai_tool

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
DEFAULT_BASE_URL = "https://gen.pollinations.ai"
INT53_MAX = 9007199254740991

# ---------------------------------------------------------------------------
# Request schema (subset alineado a Pollinations OpenAPI, api.json)
# ---------------------------------------------------------------------------

TextModelId = Literal[
    "openai",
    "openai-fast",
    "openai-large",
    "qwen-coder",
    "mistral",
    "openai-audio",
    "gemini",
    "gemini-fast",
    "deepseek",
    "grok",
    "gemini-search",
    "chickytutor",
    "midijourney",
    "claude-fast",
    "claude",
    "claude-large",
    "perplexity-fast",
    "perplexity-reasoning",
    "kimi",
    "gemini-large",
    "gemini-legacy",
    "nova-fast",
    "glm",
    "minimax",
    "nomnom",
]

Modality = Literal["text", "audio"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

VoiceId = Literal[
    "alloy",
    "echo",
    "fable",
    "onyx",
    "shimmer",
    "coral",
    "verse",
    "ballad",
    "ash",
    "sage",
    "amuch",
    "dan",
]

AudioFormat = Literal["wav", "mp3", "flac", "opus", "pcm16"]

FloatPenalty = Annotated[float, Field(ge=-2, le=2)]
FloatRepetitionPenalty = Annotated[float, Field(ge=0, le=2)]
FloatTemperature = Annotated[float, Field(ge=0, le=2)]
FloatTopP = Annotated[float, Field(ge=0, le=1)]
Int0ToInt53 = Annotated[int, Field(ge=0, le=INT53_MAX)]
SeedInt = Annotated[int, Field(ge=-1, le=INT53_MAX)]
TopLogprobsInt = Annotated[int, Field(ge=0, le=20)]
BiasValue = Annotated[int, Field(ge=-INT53_MAX, le=INT53_MAX)]


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    voice: VoiceId
    format: AudioFormat


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_usage: Optional[bool] = None


class ThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["enabled", "disabled"] = "disabled"
    budget_tokens: Optional[Annotated[int, Field(ge=1, le=INT53_MAX)]] = None


class ResponseFormatText(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["text"]


class ResponseFormatJsonObject(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["json_object"]


class ResponseFormatJsonSchemaObject(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    description: Optional[str] = None
    name: Optional[str] = None
    schema_: dict[str, Any] = Field(alias="schema")
    strict: Optional[bool] = Field(default=False)


class ResponseFormatJsonSchema(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["json_schema"]
    json_schema: ResponseFormatJsonSchemaObject


ResponseFormat = Union[ResponseFormatText, ResponseFormatJsonSchema, ResponseFormatJsonObject]


class ToolFunction(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: Optional[bool] = Field(default=False)


class ToolFunctionTool(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["function"]
    function: ToolFunction


BuiltinToolType = Literal[
    "code_execution",
    "google_search",
    "google_maps",
    "url_context",
    "computer_use",
    "file_search",
]


class ToolBuiltinTool(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: BuiltinToolType


ToolDef = Union[ToolFunctionTool, ToolBuiltinTool]


class FunctionDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolChoiceFunctionInner(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str


class ToolChoiceFunction(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["function"]
    function: ToolChoiceFunctionInner


def _normalize_tool_choice(tc: Any) -> Any:
    # LangChain a veces usa "any" para "debe llamar alguna tool"
    if tc == "any":
        return "required"
    return tc


ToolChoice = Union[Literal["none", "auto", "required"], ToolChoiceFunction]


class FunctionCallName(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str


FunctionCall = Union[Literal["none", "auto"], FunctionCallName]


class ChatPollinationsConfig(BaseModel):
    """
    Request body para POST /v1/chat/completions (excepto `messages`).

    Alineamiento con proveedor: evitar enviar campos a menos que esté configurado explícitamente.
    """

    model_config = ConfigDict(extra="forbid")

    model: Optional[TextModelId] = None
    modalities: Optional[list[Modality]] = None
    audio: Optional[AudioConfig] = None

    temperature: Optional[FloatTemperature] = None
    top_p: Optional[FloatTopP] = None
    max_tokens: Optional[Int0ToInt53] = None
    stop: Optional[Union[str, Annotated[list[str], Field(min_length=1, max_length=4)]]] = None
    seed: Optional[SeedInt] = None

    presence_penalty: Optional[FloatPenalty] = None
    frequency_penalty: Optional[FloatPenalty] = None
    repetition_penalty: Optional[FloatRepetitionPenalty] = None
    logit_bias: Optional[dict[str, BiasValue]] = None

    logprobs: Optional[bool] = None
    top_logprobs: Optional[TopLogprobsInt] = None

    stream: Optional[bool] = None
    stream_options: Optional[StreamOptions] = None

    response_format: Optional[ResponseFormat] = None

    tools: Optional[list[ToolDef]] = None
    tool_choice: Optional[ToolChoice] = None
    parallel_tool_calls: Optional[bool] = None

    user: Optional[str] = None

    functions: Optional[Annotated[list[FunctionDef], Field(min_length=1, max_length=128)]] = None
    function_call: Optional[FunctionCall] = None

    thinking: Optional[ThinkingConfig] = None
    reasoning_effort: Optional[ReasoningEffort] = None
    thinking_budget: Optional[Int0ToInt53] = None


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _extract_text_from_content_blocks(obj: Any) -> str:
    if not isinstance(obj, list):
        return ""
    parts: list[str] = []
    for block in obj:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            txt = block.get("text")
            if isinstance(txt, str) and txt:
                parts.append(txt)
    return "".join(parts)


def _usage_metadata_from_usage(usage: Any) -> UsageMetadata | None:
    if not isinstance(usage, dict):
        return None

    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")

    try:
        prompt_i = int(prompt)
        completion_i = int(completion)
        total_i = int(total)
    except Exception:
        return None

    md: UsageMetadata = {
        "input_tokens": max(prompt_i, 0),
        "output_tokens": max(completion_i, 0),
        "total_tokens": max(total_i, 0),
    }

    in_details = usage.get("prompt_tokens_details")
    out_details = usage.get("completion_tokens_details")
    if isinstance(in_details, dict):
        md["input_token_details"] = cast(Any, in_details)
    if isinstance(out_details, dict):
        md["output_token_details"] = cast(Any, out_details)

    return md


def _response_metadata_from_response(obj: dict[str, Any]) -> dict[str, Any]:
    md: dict[str, Any] = {}
    for k in ("id", "model", "created", "system_fingerprint", "user_tier", "object"):
        if k in obj and obj[k] is not None:
            md[k] = obj[k]
    citations = obj.get("citations")
    if isinstance(citations, list):
        md["citations"] = citations
    prompt_filter_results = obj.get("prompt_filter_results")
    if prompt_filter_results is not None:
        md["prompt_filter_results"] = prompt_filter_results
    return md


def _message_content_from_message_dict(message: dict[str, Any]) -> Any:
    """
    Conservar las formas multimodales cuando estén presentes:
    - message.content: str | list[part] | None
    - message.content_blocks: list[block] | None
    - message.audio.transcript: str
    """
    if "content" in message and message.get("content") is not None:
        return message.get("content")

    blocks = message.get("content_blocks")
    if isinstance(blocks, list) and blocks:
        return blocks

    audio = message.get("audio")
    if isinstance(audio, dict):
        transcript = audio.get("transcript")
        if isinstance(transcript, str) and transcript:
            return transcript

    return ""


def _delta_content_from_delta_dict(delta: dict[str, Any]) -> Any:
    if "content" in delta and delta.get("content") is not None:
        return delta.get("content")

    blocks = delta.get("content_blocks")
    if isinstance(blocks, list) and blocks:
        return blocks

    audio = delta.get("audio")
    if isinstance(audio, dict):
        transcript = audio.get("transcript")
        if isinstance(transcript, str) and transcript:
            return transcript

    return ""


def _text_from_any_content(content: Any) -> str:
    """
    Usado solo para streaming de tokens.
    Se mantienen tokens textuales en AIMessageChunk.content y se preservan partes estructuradas en additional_kwargs.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _extract_text_from_content_blocks(content)
    return ""


def _tool_call_chunks_from_delta(delta: dict[str, Any]) -> list[ToolCallChunk]:
    raw = delta.get("tool_calls")
    if not isinstance(raw, list):
        return []

    chunks: list[ToolCallChunk] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        idx = item.get("index")
        fn = item.get("function")
        if not isinstance(idx, int) or not isinstance(fn, dict):
            continue

        name = fn.get("name")
        args = fn.get("arguments")

        chunks.append(
            tool_call_chunk(
                name=name if isinstance(name, str) else None,
                args=args if isinstance(args, str) else None,
                id=item.get("id") if isinstance(item.get("id"), str) else None,
                index=idx,
            )
        )

    return chunks


def _iter_sse_json_events_sync(resp: Any) -> Iterator[dict[str, Any]]:
    """
    SSE parser para respuestas streaming de httpx:
    - Admite varias líneas "data:" por evento.
    - Trata una línea en blanco como delimitador cuando está presente.
    - Analiza cada línea "data:" como un evento (comportamiento común del proveedor).
    """
    data_lines: list[str] = []

    def flush() -> Iterator[dict[str, Any]]:
        if not data_lines:
            return iter(())
        payload = "\n".join(data_lines).strip()
        data_lines.clear()
        if not payload:
            return iter(())
        if payload == "[DONE]":
            return iter(({"__done__": True},))
        try:
            obj = json.loads(payload)
        except Exception:
            return iter(())
        if isinstance(obj, dict):
            return iter((obj,))
        return iter(())

    for raw_line in resp.iter_lines():
        if raw_line is None:
            continue
        if not isinstance(raw_line, str):
            try:
                line = raw_line.decode("utf-8", "ignore")
            except Exception:
                continue
        else:
            line = raw_line

        if line == "":
            yield from flush()
            continue

        if not line.startswith("data:"):
            continue

        piece = line[len("data:") :].lstrip()
        if piece == "[DONE]":
            yield {"__done__": True}
            return

        # La mayoría de los proveedores envían un JSON por línea de datos; aun así, admitimos la unión.
        data_lines.append(piece)
        if len(data_lines) == 1:
            # Tratar de parsear líneas solas inmediatamente (ruta rápida).
            for evt in flush():
                yield evt

    yield from flush()


async def _iter_sse_json_events_async(resp: Any) -> AsyncIterator[dict[str, Any]]:
    data_lines: list[str] = []

    async def flush() -> list[dict[str, Any]]:
        if not data_lines:
            return []
        payload = "\n".join(data_lines).strip()
        data_lines.clear()
        if not payload:
            return []
        if payload == "[DONE]":
            return [{"__done__": True}]
        try:
            obj = json.loads(payload)
        except Exception:
            return []
        return [obj] if isinstance(obj, dict) else []

    async for line in resp.aiter_lines():
        if line == "":
            for evt in await flush():
                yield evt
            continue

        if not line.startswith("data:"):
            continue

        piece = line[len("data:") :].lstrip()
        if piece == "[DONE]":
            yield {"__done__": True}
            return

        data_lines.append(piece)
        if len(data_lines) == 1:
            for evt in await flush():
                yield evt

    for evt in await flush():
        yield evt

def _parse_tool_calls(message: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return [], []

    tool_calls: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function":
            continue

        fn = item.get("function")
        if not isinstance(fn, dict):
            continue

        name = fn.get("name")
        args_s = fn.get("arguments")
        tc_id = item.get("id")

        if not isinstance(name, str) or not name:
            continue

        args: dict[str, Any] = {}
        if isinstance(args_s, str) and args_s.strip():
            try:
                loaded = json.loads(args_s)
                if isinstance(loaded, dict):
                    args = loaded
            except Exception as e:
                invalid.append(
                    {"name": name, "args": {}, "id": tc_id, "type": "invalid_tool_call", "error": str(e)}
                )
                continue

        tool_calls.append({"name": name, "args": args, "id": tc_id, "type": "tool_call"})

    return tool_calls, invalid


# ------------------------------------------------------------------------------------
# Chat wrapper (alineado con LangChain v1.2.x, streaming real via .stream/.astream)
# ------------------------------------------------------------------------------------


class ChatPollinations(BaseChatModel):
    """
    LangChain ChatModel para OpenAI-compatible endpoint de Pollinations: /v1/chat/completions

    Contrato para el Streaming:
    - .invoke/.ainvoke usan respuesta no-streaming
    - .stream/.astream realizan streaming real and emiten message chunks
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 120.0

    request_defaults: ChatPollinationsConfig = Field(default_factory=ChatPollinationsConfig)

    # Solicitar uso en el stream (el proveedor admite stream_options.include_usage)
    include_usage_in_stream: bool = True

    # Conservar el delta multimodal estructurado en additional_kwargs mientras se siguen transmitiendo tokens.
    preserve_multimodal_deltas: bool = True

    _http: PollinationsHttpClient = PrivateAttr()

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 120.0,
        request_defaults: ChatPollinationsConfig | None = None,
        include_usage_in_stream: bool = True,
        preserve_multimodal_deltas: bool = True,
        **kwargs: Any,
    ) -> None:
        rd_keys = set(ChatPollinationsConfig.model_fields.keys())
        rd_kwargs = {k: v for k, v in kwargs.items() if k in rd_keys}
        lc_kwargs = {k: v for k, v in kwargs.items() if k not in rd_keys}

        if request_defaults is not None and rd_kwargs:
            raise ValueError("Do not mix request_defaults=... with loose request body parameters.")

        rd = request_defaults or ChatPollinationsConfig(**rd_kwargs)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout_s=timeout_s,
            request_defaults=rd,
            include_usage_in_stream=include_usage_in_stream,
            preserve_multimodal_deltas=preserve_multimodal_deltas,
            **lc_kwargs,
        )

        auth = AuthConfig.from_env_or_value(self.api_key)
        self._http = PollinationsHttpClient(
            config=HttpConfig(base_url=self.base_url, timeout_s=self.timeout_s),
            api_key=auth.api_key,
        )

    @property
    def _llm_type(self) -> str:
        return "pollinations-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.request_defaults.model or "",
            "base_url": self.base_url,
            "timeout_s": self.timeout_s,
            "include_usage_in_stream": self.include_usage_in_stream,
            "preserve_multimodal_deltas": self.preserve_multimodal_deltas,
        }

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: Any | None = None,
        parallel_tool_calls: bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> "ChatPollinations":
        if kwargs:
            raise ValueError("No extra fields are allowed in bind_tools().")

        # tools puede contener:
        # - langchain tools (@tool, BaseTool, dict con function)
        # - builtin tools de Pollinations ({ "type": "url_context" }, etc.)
        openai_tools: list[dict[str, Any]] = []

        for t in tools:
            if isinstance(t, dict) and "type" in t and t.get("type") in {
                "code_execution",
                "google_search",
                "google_maps",
                "url_context",
                "computer_use",
                "file_search",
            }:
                # builtin tool, ya está en el formato correcto
                openai_tools.append(t)
            else:
                # tool de función normal → formato OpenAI {"type":"function","function":{...}}
                td = tool_to_openai_tool(t)
                if strict is not None:
                    fn = td.get("function")
                    if isinstance(fn, dict):
                        fn["strict"] = strict
                openai_tools.append(td)

        # Validar/convertir dict -> modelo Pydantic (Union)
        adapter = TypeAdapter(ToolDef)
        tool_defs = [adapter.validate_python(t) for t in openai_tools]

        # NO usar ToolDef.model_validate: ToolDef es solo un Union typing.
        # Pydantic validará al hacer .model_dump() en ChatPollinationsConfig.
        rd = self.request_defaults.model_copy(deep=True)
        rd.tools = tool_defs  # type: ignore[assignment]

        if tool_choice is not None:
            tool_choice = _normalize_tool_choice(tool_choice)
            rd.tool_choice = TypeAdapter(ToolChoice).validate_python(tool_choice)  # valida y tipa correctamente

        if parallel_tool_calls is not None:
            rd.parallel_tool_calls = parallel_tool_calls  # type: ignore[assignment]

        return ChatPollinations(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_s=self.timeout_s,
            request_defaults=rd,
            include_usage_in_stream=self.include_usage_in_stream,
            preserve_multimodal_deltas=self.preserve_multimodal_deltas,
        )

    def _build_payload(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Construye el request payload.

        Importante: Ignora los kwargs ajenos al proveedor y que la capa Runnable pueda pasar.
        (p.e., stream_mode=..., include_names=..., etc.)
        """
        payload: dict[str, Any] = {"messages": lc_messages_to_openai(messages)}

        payload.update(
            self.request_defaults.model_dump(
                by_alias=True,
                exclude_none=True,
                exclude_unset=True,
            )
        )

        if "tool_choice" in payload:
            payload["tool_choice"] = _normalize_tool_choice(payload.get("tool_choice"))

        if stop is not None:
            payload["stop"] = stop

        provider_kwargs: dict[str, Any] = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if k == "stop":
                if stop is None:
                    provider_kwargs["stop"] = v
                continue
            if k in ChatPollinationsConfig.model_fields:
                provider_kwargs[k] = v

        if "tool_choice" in provider_kwargs:
            provider_kwargs["tool_choice"] = _normalize_tool_choice(provider_kwargs.get("tool_choice"))

        if provider_kwargs:
            validated = ChatPollinationsConfig(**provider_kwargs)
            payload.update(
                validated.model_dump(
                    by_alias=True,
                    exclude_none=True,
                    exclude_unset=True,
                )
            )

        if "tool_choice" in payload:
            payload["tool_choice"] = _normalize_tool_choice(payload.get("tool_choice"))

        return payload

    def _parse_chat_result(self, data: dict[str, Any]) -> ChatResult:
        response_metadata = _response_metadata_from_response(data)
        usage_metadata = _usage_metadata_from_usage(data.get("usage"))

        choices = data.get("choices", [])
        generations: list[ChatGeneration] = []

        if isinstance(choices, list):
            for ch in choices:
                if not isinstance(ch, dict):
                    continue
                message = ch.get("message") or {}
                if not isinstance(message, dict):
                    message = {}

                content = _message_content_from_message_dict(message)

                additional_kwargs: dict[str, Any] = {}
                for k in ("tool_calls", "function_call", "audio", "reasoning_content", "content_blocks"):
                    if k in message and message[k] is not None:
                        additional_kwargs[k] = message[k]

                tool_calls, invalid_tool_calls = _parse_tool_calls(message)

                msg = AIMessage(
                    content=content,
                    additional_kwargs=additional_kwargs,
                    response_metadata=response_metadata,
                    usage_metadata=usage_metadata,
                    tool_calls=tool_calls,
                    invalid_tool_calls=invalid_tool_calls,
                )

                gen_info: dict[str, Any] = {}
                if ch.get("finish_reason") is not None:
                    gen_info["finish_reason"] = ch.get("finish_reason")
                if ch.get("logprobs") is not None:
                    gen_info["logprobs"] = ch.get("logprobs")
                if ch.get("content_filter_results") is not None:
                    gen_info["content_filter_results"] = ch.get("content_filter_results")
                if isinstance(ch.get("index"), int):
                    gen_info["index"] = ch.get("index")

                generations.append(ChatGeneration(message=msg, generation_info=gen_info or None))

        if not generations:
            msg = AIMessage(content="", response_metadata=response_metadata, usage_metadata=usage_metadata)
            generations = [ChatGeneration(message=msg)]

        # Mantener llm_output al mínimo para evitar sobrecargar la propagación de response_metadata.
        llm_output: dict[str, Any] = {
            "model_name": data.get("model") or (self.request_defaults.model or ""),
            "system_fingerprint": data.get("system_fingerprint"),
            "token_usage": data.get("usage"),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, **kwargs)
        resp = self._http.post_json(CHAT_COMPLETIONS_PATH, payload, stream=False)
        return self._parse_chat_result(resp.json())

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, **kwargs)
        resp = await self._http.apost_json(CHAT_COMPLETIONS_PATH, payload, stream=False)
        return self._parse_chat_result(resp.json())

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop, **kwargs)
        payload["stream"] = True

        if self.include_usage_in_stream:
            so = payload.get("stream_options")
            if so is None:
                payload["stream_options"] = {"include_usage": True}
            elif isinstance(so, dict) and "include_usage" not in so:
                so2 = dict(so)
                so2["include_usage"] = True
                payload["stream_options"] = so2

        pending_usage_md: UsageMetadata | None = None
        pending_usage_response_md: dict[str, Any] | None = None
        emitted_final_usage = False

        def emit_final_usage_once() -> Iterator[ChatGenerationChunk]:
            nonlocal emitted_final_usage
            if emitted_final_usage:
                return iter(())
            if pending_usage_md is None:
                return iter(())
            emitted_final_usage = True
            msg_chunk = AIMessageChunk(
                content="",
                response_metadata=pending_usage_response_md or {},
                usage_metadata=pending_usage_md,
            )
            return iter((ChatGenerationChunk(message=msg_chunk, generation_info={"usage": True}),))

        with self._http.stream_post_json(CHAT_COMPLETIONS_PATH, payload) as r:
            self._http.raise_for_status(r)

            for evt in _iter_sse_json_events_sync(r):
                if evt.get("__done__") is True:
                    break

                # Monitorear el uso más reciente, pero sin adjuntarlo a chunks normales (evitar el conteo doble).
                if "usage" in evt and evt.get("usage") is not None:
                    usage_md = _usage_metadata_from_usage(evt.get("usage"))
                    if usage_md is not None:
                        pending_usage_md = usage_md
                        pending_usage_response_md = _response_metadata_from_response(evt)

                response_metadata = _response_metadata_from_response(evt)

                choices = evt.get("choices") or []
                if not isinstance(choices, list) or not choices:
                    # Algunos proveedores pueden transmitir eventos de solo uso; ignorarlos aquí para evitar duplicados.
                    continue

                # Para máxima compatibilidad con la agregación LangChain, transmita solo la primera opción.
                choice0 = choices[0]
                if not isinstance(choice0, dict):
                    continue

                delta = choice0.get("delta") or {}
                if not isinstance(delta, dict):
                    continue

                delta_content = _delta_content_from_delta_dict(delta)
                text = _text_from_any_content(delta_content)

                tool_call_chunks = _tool_call_chunks_from_delta(delta)

                additional_kwargs: dict[str, Any] = {}
                # Preservar contenido estructurado para consumidores multimodales sin interrumpir el stream.
                if self.preserve_multimodal_deltas and isinstance(delta_content, list):
                    additional_kwargs["content_parts"] = delta_content

                for k in ("tool_calls", "function_call", "audio", "reasoning_content", "content_blocks"):
                    if k in delta and delta[k] is not None:
                        additional_kwargs[k] = delta[k]

                msg_chunk = AIMessageChunk(
                    content=text,
                    additional_kwargs=additional_kwargs,
                    tool_call_chunks=tool_call_chunks,
                    response_metadata=response_metadata,
                    usage_metadata=None,  # importante: evitar suma repetida de uso durante la agregación de chunks
                )

                gen_info: dict[str, Any] = {}
                if choice0.get("finish_reason") is not None:
                    gen_info["finish_reason"] = choice0.get("finish_reason")
                if choice0.get("logprobs") is not None:
                    gen_info["logprobs"] = choice0.get("logprobs")
                if choice0.get("content_filter_results") is not None:
                    gen_info["content_filter_results"] = choice0.get("content_filter_results")
                if isinstance(choice0.get("index"), int):
                    gen_info["index"] = choice0.get("index")

                yield ChatGenerationChunk(message=msg_chunk, generation_info=gen_info or None)

            # Emitir uso exactamente una vez al final.
            yield from emit_final_usage_once()

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop, **kwargs)
        payload["stream"] = True

        if self.include_usage_in_stream:
            so = payload.get("stream_options")
            if so is None:
                payload["stream_options"] = {"include_usage": True}
            elif isinstance(so, dict) and "include_usage" not in so:
                so2 = dict(so)
                so2["include_usage"] = True
                payload["stream_options"] = so2

        pending_usage_md: UsageMetadata | None = None
        pending_usage_response_md: dict[str, Any] | None = None
        emitted_final_usage = False

        async def emit_final_usage_once() -> None:
            nonlocal emitted_final_usage
            if emitted_final_usage or pending_usage_md is None:
                return
            emitted_final_usage = True
            msg_chunk = AIMessageChunk(
                content="",
                response_metadata=pending_usage_response_md or {},
                usage_metadata=pending_usage_md,
            )
            yield_chunk = ChatGenerationChunk(message=msg_chunk, generation_info={"usage": True})
            # Utilizar un truco de generador asíncrono: el caller espera los yields de esta función.
            # Esta función solo se llama al final; el yield retorna un solo elemento vía outer scope.
            return yield_chunk  # type: ignore[return-value]

        async with self._http.astream_post_json(CHAT_COMPLETIONS_PATH, payload) as r:
            self._http.raise_for_status(r)

            async for evt in _iter_sse_json_events_async(r):
                if evt.get("__done__") is True:
                    break

                if "usage" in evt and evt.get("usage") is not None:
                    usage_md = _usage_metadata_from_usage(evt.get("usage"))
                    if usage_md is not None:
                        pending_usage_md = usage_md
                        pending_usage_response_md = _response_metadata_from_response(evt)

                response_metadata = _response_metadata_from_response(evt)

                choices = evt.get("choices") or []
                if not isinstance(choices, list) or not choices:
                    continue

                choice0 = choices[0]
                if not isinstance(choice0, dict):
                    continue

                delta = choice0.get("delta") or {}
                if not isinstance(delta, dict):
                    continue

                delta_content = _delta_content_from_delta_dict(delta)
                text = _text_from_any_content(delta_content)

                tool_call_chunks = _tool_call_chunks_from_delta(delta)

                additional_kwargs: dict[str, Any] = {}
                if self.preserve_multimodal_deltas and isinstance(delta_content, list):
                    additional_kwargs["content_parts"] = delta_content

                for k in ("tool_calls", "function_call", "audio", "reasoning_content", "content_blocks"):
                    if k in delta and delta[k] is not None:
                        additional_kwargs[k] = delta[k]

                msg_chunk = AIMessageChunk(
                    content=text,
                    additional_kwargs=additional_kwargs,
                    tool_call_chunks=tool_call_chunks,
                    response_metadata=response_metadata,
                    usage_metadata=None,  # importante: evitar suma de uso repetida durante la agregación del chunk
                )

                gen_info: dict[str, Any] = {}
                if choice0.get("finish_reason") is not None:
                    gen_info["finish_reason"] = choice0.get("finish_reason")
                if choice0.get("logprobs") is not None:
                    gen_info["logprobs"] = choice0.get("logprobs")
                if choice0.get("content_filter_results") is not None:
                    gen_info["content_filter_results"] = choice0.get("content_filter_results")
                if isinstance(choice0.get("index"), int):
                    gen_info["index"] = choice0.get("index")

                yield ChatGenerationChunk(message=msg_chunk, generation_info=gen_info or None)

            if not emitted_final_usage and pending_usage_md is not None:
                emitted_final_usage = True
                msg_chunk = AIMessageChunk(
                    content="",
                    response_metadata=pending_usage_response_md or {},
                    usage_metadata=pending_usage_md,
                )
                yield ChatGenerationChunk(message=msg_chunk, generation_info={"usage": True})
