# Code Structure and Composition (langchain-pollinations)

## 1) Module Map

The library is organized within the `langchain_pollinations` package, exposing a concise public API and delegating complex logic to internal support modules.

```text
langchain_pollinations/
  __init__.py              -> Exports the public API: main classes, account types,
                              audio response types, and error types.
  chat.py                  -> ChatPollinations (BaseChatModel), ChatPollinationsConfig, dynamic
                              text model catalog, SSE helpers, and response parsing.
  image.py                 -> ImagePollinations (BaseModel + Runnable), ImagePromptParams,
                              and dynamic image model catalog.
  tts.py                   -> TTSPollinations (BaseModel + Runnable), SpeechRequest,
                              AudioFormat, VoiceId, and TTS request building.
  stt.py                   -> STTPollinations (BaseModel + Runnable), TranscriptionParams,
                              TranscriptionResponse, TranscriptionFormat, AudioInputFormat,
                              multipart construction and response parsing.
  account.py               -> AccountInformation (dataclass), Pydantic response models
                              (AccountProfile, AccountUsageRecord, AccountUsageResponse), and
                              query parameters (AccountUsageParams, AccountUsageDailyParams).
  models.py                -> ModelInformation (dataclass), lists text, image, and audio models.
  _audio_catalog.py        -> Shared audio model catalog: module-level cache, threading
                              primitives, _load_audio_model_ids(), and AudioModelId alias.
                              Used by both tts.py and stt.py.
  _auth.py                 -> AuthConfig (frozen dataclass): API key resolution from argument
                              or environment variable.
  _client.py               -> PollinationsHttpClient: httpx sync/async wrapper, structured
                              error handling, JSON POST, multipart POST, GET, streaming,
                              and logging with credential redaction.
  _errors.py               -> PollinationsError (base) and PollinationsAPIError (slots dataclass):
                              exception hierarchy with structured fields from the JSON envelope.
  _sse.py                  -> Minimalist Server-Sent Events parser for text blocks.
                              Exposes SSEEvent and iter_sse_events_from_text.
  _openai_compat.py        -> Request/response TypedDicts, normalization for multimodal
                              content parts (audio, video, file, thinking), and conversion
                              of LangChain messages → OpenAI, tool_to_openai_tool.
```

## 2) Public vs. Internal API

The public surface is defined in `__init__.py` via `__all__`:

```python
__all__ = [
    # Main classes
    "ChatPollinations",
    "ImagePollinations",
    "TTSPollinations",
    "STTPollinations",
    "AccountInformation",
    "ModelInformation",
    # Account types (exported to facilitate response typing)
    "AccountProfile",
    "AccountTier",
    "AccountUsageRecord",
    "AccountUsageResponse",
    # Audio response and format types (exported to facilitate response typing)
    "AudioInputFormat",
    "TranscriptionFormat",
    "TranscriptionResponse",
    # Exceptions
    "PollinationsAPIError",
]
```

Modules with an underscore prefix (`_client.py`, `_auth.py`, `_sse.py`, `_openai_compat.py`, `_errors.py`, `_audio_catalog.py`) are exclusively for internal use. They encapsulate HTTP transport, authentication, SSE parsing, multimodal content normalization, the error hierarchy, and the shared audio model catalog. They are invisible to the end user by naming convention and are absent from `__all__`.

## 3) Composition by Responsibilities

### 3.1 Chat (`chat.py` + `_openai_compat.py`)

**`chat.py`** contains two distinct responsibilities:

**Configuration and Chat Model:**
- `ChatPollinationsConfig` — Pydantic model (`extra="forbid"`) with all parameters for the `/v1/chat/completions` endpoint: `model`, `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `modalities`, `audio`, `tools`, `tool_choice`, `response_format`, `thinking`, `reasoning_effort`, `thinking_budget`, `stream_options`, and legacy variants (`functions`, `function_call`).
- `ChatPollinations` — inherits from `BaseChatModel`. It accepts `ChatPollinationsConfig` fields as direct kwargs in the constructor or grouped in `request_defaults=ChatPollinationsConfig(...)`. It implements `_generate` (sync), `_stream` (sync), `_agenerate` (async), `_astream` (async), `bind_tools`, and `with_structured_output`.

**Dynamic Text Model Catalog:**
- `_load_text_model_ids(api_key, force)` — loads text model IDs from the `/text/models` endpoint the first time `ChatPollinations` is instantiated in a process. It implements double-checked locking with `threading.Lock` to be thread-safe. If the call fails, it keeps the static fallback `_FALLBACK_TEXT_MODEL_IDS` without raising an exception. The `model` `field_validator` emits a `UserWarning` (without blocking the request) if the provided ID is not in the loaded catalog.

**Parsing Helpers in `chat.py`:**

| Function | Responsibility |
|---|---|
| `_iter_sse_json_events_sync(resp)` | Parses the SSE stream line-by-line synchronously, producing JSON dicts. |
| `_iter_sse_json_events_async(resp)` | Async version of the SSE parser, using `aiter_lines()`. |
| `_message_content_from_message_dict(message)` | Extracts message content from a full response: prioritizes `content`, then `content_blocks`, then `audio.transcript`. |
| `_delta_content_from_delta_dict(delta)` | Extracts incremental content from a streaming delta: `content` string or list, `content_blocks`, or `audio.transcript`. |
| `_extract_text_from_content_blocks(obj)` | Concatenates only `"type": "text"` blocks from a content blocks list, omitting thinking. |
| `_usage_metadata_from_usage(usage)` | Converts the response `usage` field to LangChain's `UsageMetadata`, including `input_token_details` and `output_token_details`. |
| `_response_metadata_from_response(obj)` | Extracts general response metadata: `id`, `model`, `created`, `system_fingerprint`, `user_tier`, `citations`, `prompt_filter_results`. |
| `_parse_tool_calls(message)` | Extracts and validates tool calls from a message, separating valid (`tool_calls`) from invalid (`invalid_tool_calls`). |
| `_normalize_tool_choice(tc)` | Converts LangChain's `tool_choice` value to the format expected by the API (e.g., `"any"` → `"required"`). |
| `_text_from_any_content(content)` | Converts any content representation (str or block list) to a plain string for token streaming. |
| `_tool_call_chunks_from_delta(delta)` | Parses tool call deltas in streaming and converts them to LangChain `ToolCallChunk`. |

**`_openai_compat.py`** acts as a translation and typing layer:

- **Input Normalization**: `lc_messages_to_openai(messages)` converts LangChain's `BaseMessage` list to OpenAI-compatible dicts. Each message type is handled specifically:
  - `ToolMessage` → extracts only the text (`_extract_text_from_parts`) and omits the `name` field (provider incompatibility).
  - `AIMessage` → converts `tool_calls` and `invalid_tool_calls` to OpenAI format using `_lc_tool_call_to_openai_tool_call`.
  - All messages pass through `_normalize_message_content`, which iterates through each part with `_normalize_content_part`.
- **Multimodal Part Normalization**:
  - `_normalize_input_audio_part(part)` — normalizes audio variants (flat, `base64`, `mime_type`, `audio` sub-dict) to the canonical form `{type: "input_audio", input_audio: {data, format}}`.
  - `_normalize_video_url_part(part)` — accepts the canonical form (`video_url.url`) and flat variant (`url` in root).
  - `_normalize_file_part(part)` — accepts the canonical form (`file` sub-dict) and flat variant, preserving `cache_control`.
  - `_normalize_content_part(part)` — central dispatcher that removes the `id` field (rejected by the API) and routes each type to its specific normalizer.
  - `_infer_audio_format_from_mime(mime_type)` — maps MIME types to allowed audio formats: `wav`, `mp3`, `flac`, `opus`, `pcm16`.
- **Tool Conversion**: `tool_to_openai_tool(tool)` transforms a LangChain tool definition into the schema `{type: "function", function: {...}}`. It supports: dicts already in OpenAI format (pass-through), LangChain's `convert_to_openai_tool` (main route), `args_schema`, `tool_call_schema`, `get_input_schema()`, `BaseModel`, and `TypedDict` (via `TypeAdapter`).
- **Request TypedDicts**: `ContentBlockText`, `ContentBlockImageUrl`, `ContentBlockInputAudio`, `ContentBlockVideoUrl`, `ContentBlockFile`, `ContentBlockThinking`, `ContentBlockRedactedThinking`, and the `ContentBlock` union.
- **Response TypedDicts**: `AudioTranscript`, `ChatCompletionResponse`, `ContentFilterDetail`, `ContentFilterResult`, `PromptFilterResultItem`.
- **Literal Types**: `UserTier = Literal["anonymous", "spore", "seed", "flower", "nectar"]`.

### 3.2 Image (`image.py`)

**Dynamic Image Model Catalog:**
- `_load_image_model_ids(api_key, force)` — structured identically to the chat version: double-checked locking, `_FALLBACK_IMAGE_MODEL_IDS` static fallback, fails silently. It is invoked in `ImagePollinations.__init__` before Pydantic runs `field_validators`, ensuring `_validate_model_id` has access to the updated catalog.

**`ImagePromptParams`** — Pydantic model (`extra="forbid"`, `populate_by_name=True`) that validates and serializes query parameters for the `GET /image/{prompt}` endpoint:

| Field | API Alias | Default | Description |
|---|---|---|---|
| `model` | — | `"zimage"` | Model ID. |
| `width` | — | `1024` | Width in pixels (`≥ 0`). |
| `height` | — | `1024` | Height in pixels (`≥ 0`). |
| `seed` | — | `0` | Random seed (`-1` to `2147483647`). |
| `enhance` | — | `False` | Prompt enhancement. |
| `negative_prompt` | — | `"worst quality, blurry"` | Negative prompt. |
| `safe` | — | `False` | Content filter. |
| `quality` | — | `"medium"` | `"low"/"medium"/"high"/"hd"` (gptimage). |
| `image` | — | `None` | Reference image URLs (separated by `,` or `\|`). |
| `transparent` | — | `False` | Transparent background (gptimage). |
| `duration` | — | `None` | Video duration in seconds (`1`–`10`). |
| `aspect_ratio` | `aspectRatio` | `None` | Aspect ratio (e.g., `"16:9"`). |
| `audio` | — | `False` | Include audio track in video (veo). |

`to_query()` calls `model_dump(by_alias=True, exclude_none=True)` to produce the query string dict.

**`ImagePollinations`** — inherits from `BaseModel` + `Runnable[str, bytes]` (`arbitrary_types_allowed=True`, `extra="forbid"`, `populate_by_name=True`). All `ImagePromptParams` fields are present as `Optional` with a `None` default (the effective default is set by `ImagePromptParams` when building the query). The HTTP client is stored in `_http: PollinationsHttpClient = PrivateAttr()`.

Public methods:

| Method | Type | Return | Description |
|---|---|---|---|
| `generate(prompt, *, params, **kwargs)` | sync | `bytes` | Generates and returns bytes. |
| `agenerate(prompt, *, params, **kwargs)` | async | `bytes` | Async version. |
| `generate_response(prompt, *, params, **kwargs)` | sync | `httpx.Response` | Returns the raw HTTP response. |
| `agenerate_response(prompt, *, params, **kwargs)` | async | `httpx.Response` | Async version. |
| `invoke(input, config, **kwargs)` | sync | `bytes` | LangChain Runnable compatibility. |
| `ainvoke(input, config, **kwargs)` | async | `bytes` | Async version. |
| `with_params(**overrides)` | — | `ImagePollinations` | Returns a new instance with merged config without mutating the original. `api_key` is manually re-injected since it is `exclude=True`. |

### 3.3 Account and Models (`account.py`, `models.py`)

**`account.py`** exposes two layers:

*Pydantic Data Models* (exported from `__init__.py`):
- `AccountProfile` (`extra="allow"`, `populate_by_name=True`) — maps API camelCase (`githubUsername` → `github_username`, `createdAt` → `created_at`, `nextResetAt` → `next_reset_at`). The `tier` field accepts `AccountTier`; new backend fields are preserved without `ValidationError`.
- `AccountUsageRecord` (`extra="allow"`) — a usage record. Token counters are `float` (the spec declares them as `number`). `api_key_type` and `meter_source` are open `str` to tolerate future values.
- `AccountUsageResponse` (`extra="allow"`) — wraps `usage: list[AccountUsageRecord]` and `count: int`.

*Query Parameters* (in `account.py`, not exported from `__init__`):
- `AccountUsageParams` (`extra="forbid"`) — `format`, `limit` (`1`–`50000`), `before` (ISO cursor).
- `AccountUsageDailyParams` (`extra="forbid"`) — only `format`.

*`AccountInformation`* (`@dataclass(slots=True)`) — account client. `__post_init__` initializes `PollinationsHttpClient` with `AuthConfig`. The static method `_parse_response(response)` discriminates JSON vs. CSV via the `Content-Type` header. `get_profile()` and `aget_profile()` return `AccountProfile` (via `model_validate`). `get_usage()` and `aget_usage()` return `AccountUsageResponse` for JSON and `str` for CSV.

**`models.py`** — `ModelInformation` (`@dataclass(slots=True)`). The static method `_extract_model_ids(models_data)` extracts IDs from items searching in order: `id`, `model`, `name`. `get_available_models()` and `aget_available_models()` orchestrate parallel calls to text, image, and audio model endpoints, catching individual exceptions to return the partial set that is available. The returned dict has three keys: `"text"`, `"image"`, and `"audio"`.

*Methods on `ModelInformation`*:

| Method | Endpoint | Returns |
|---|---|---|
| `list_text_models()` | `GET /text/models` | `list[dict] \| dict` |
| `list_image_models()` | `GET /image/models` | `list[dict] \| dict` |
| `list_audio_models()` | `GET /audio/models` | `list[dict] \| dict` |
| `list_compatible_models()` | `GET /v1/models` | `dict` |
| `get_available_models()` | all three above | `dict[str, list[str]]` |
| `alist_*` / `aget_*` | same, async | same returns |

### 3.4 TTS (`tts.py`)

**Module-level type aliases:**
- `AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]` — closed enum matching the API spec.
- `VoiceId = str` — open string type to tolerate future voice additions without `ValidationError`.

**`SpeechRequest`** — Pydantic model (`extra="forbid"`, `populate_by_name=True`) that mirrors the `CreateSpeechRequest` OpenAPI schema for `POST /v1/audio/speech`:

| Field | Default | Constraint | Notes |
|---|---|---|---|
| `input` | required | `minlength=1, maxlength=4096` | Text to synthesise. |
| `model` | `None` | — | `None` defers to API default. |
| `voice` | `"alloy"` | — | Open `VoiceId` type. |
| `response_format` | `"mp3"` | `AudioFormat` | Closed enum. |
| `speed` | `1.0` | `0.25`–`4.0` | Playback speed multiplier. |
| `duration` | `None` | `3.0`–`300.0` | `elevenmusic` only; omitted when `None`. |
| `instrumental` | `None` | — | `elevenmusic` only; omitted when `None`. |

`to_body()` calls `model_dump(exclude_none=True)`, silently omitting model-specific optional fields when they are `None`.

**`TTSPollinations`** — inherits from `BaseModel` + `Runnable[str, bytes]` (`arbitrary_types_allowed=True`, `extra="forbid"`, `populate_by_name=True`). All `SpeechRequest` fields except `input` are present as `Optional` with `None` defaults. The HTTP client is stored in `_http: PollinationsHttpClient = PrivateAttr()`.

Internal helpers:

| Method | Description |
|---|---|
| `_defaults_dict()` | Collects only non-`None` instance-level fields into a base dict. |
| `_build_body(text, params, kwargs)` | Three-layer merge: instance defaults → per-call `params` → per-call `kwargs`. Injects `input=text` last (cannot be overridden). Validates via `SpeechRequest` and returns `to_body()`. |

Public methods:

| Method | Type | Return | Description |
|---|---|---|---|
| `generate(text, *, params, **kwargs)` | sync | `bytes` | Synthesises text and returns audio bytes. |
| `agenerate(text, *, params, **kwargs)` | async | `bytes` | Async version. |
| `generate_response(text, *, params, **kwargs)` | sync | `httpx.Response` | Returns the raw HTTP response. |
| `agenerate_response(text, *, params, **kwargs)` | async | `httpx.Response` | Async version. |
| `invoke(input, config, **kwargs)` | sync | `bytes` | LangChain Runnable compatibility. |
| `ainvoke(input, config, **kwargs)` | async | `bytes` | Async version. |
| `with_params(**overrides)` | — | `TTSPollinations` | New instance with merged config; `api_key` re-injected. |

### 3.5 STT (`stt.py`)

**Module-level type aliases and constants:**
- `AudioInputFormat = Literal["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]` — accepted input audio formats.
- `TranscriptionFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]` — accepted output formats.
- `_AUDIO_MIME_TYPES: dict[str, str]` — maps file extensions to MIME type strings (e.g., `"mp3"` → `"audio/mpeg"`, `"wav"` → `"audio/wav"`).

**`TranscriptionResponse`** — Pydantic model (`extra="allow"`). Declares only `text: str`. When `response_format="verbose_json"`, additional Whisper fields (`language`, `duration`, `segments`, `words`) are preserved transparently in `model_extra`.

**`TranscriptionParams`** — Pydantic model (`extra="forbid"`) that carries the non-file form fields for `POST /v1/audio/transcriptions`:

| Field | Default | Constraint | Notes |
|---|---|---|---|
| `model` | `None` | — | `None` defers to `whisper-large-v3`. |
| `language` | `None` | — | ISO-639-1; `None` enables auto-detection. |
| `prompt` | `None` | — | Context hint for transcription. |
| `response_format` | `None` | `TranscriptionFormat` | `None` defers to `"json"`. |
| `temperature` | `None` | `0.0`–`1.0` | Sampling temperature. |

`to_form_data()` calls `model_dump(exclude_none=True)` and casts all values to `str` (required by httpx's multipart form interface).

**`STTPollinations`** — inherits from `BaseModel` + `Runnable[bytes, TranscriptionResponse | str]` (`arbitrary_types_allowed=True`, `extra="forbid"`, `populate_by_name=True`). All `TranscriptionParams` fields are present as `Optional` with `None` defaults. Additionally exposes `file_name: str = "audio.mp3"` as an instance-level default that determines the MIME type sent in the multipart body. The HTTP client is stored in `_http: PollinationsHttpClient = PrivateAttr()`.

Internal helpers:

| Method | Description |
|---|---|
| `_defaults_dict()` | Collects non-`None` instance fields including `file_name`. |
| `_mime_type_for(filename)` | Extracts extension, lowercases it, looks it up in `_AUDIO_MIME_TYPES`; falls back to `application/octet-stream`. |
| `_build_multipart(audio, params, kwargs)` | Three-layer merge: instance defaults → per-call `params` → per-call `kwargs`. Pops `file_name` before `TranscriptionParams` validation (`extra="forbid"` rejects unknown fields). Returns `(files_dict, form_data, params_obj)`. |
| `_parse_transcription_response(resp, response_format)` | Driven by actual `Content-Type` header. `application/json` → `TranscriptionResponse.model_validate(resp.json())`. Non-JSON → `resp.text`. |

Public methods:

| Method | Type | Return | Description |
|---|---|---|---|
| `transcribe(audio, *, params, **kwargs)` | sync | `TranscriptionResponse \| str` | Transcribes audio and returns parsed response. |
| `atranscribe(audio, *, params, **kwargs)` | async | `TranscriptionResponse \| str` | Async version. |
| `transcribe_response(audio, *, params, **kwargs)` | sync | `httpx.Response` | Returns the raw HTTP response. |
| `atranscribe_response(audio, *, params, **kwargs)` | async | `httpx.Response` | Async version. |
| `invoke(input, config, **kwargs)` | sync | `TranscriptionResponse \| str` | LangChain Runnable compatibility. |
| `ainvoke(input, config, **kwargs)` | async | `TranscriptionResponse \| str` | Async version. |
| `with_params(**overrides)` | — | `STTPollinations` | New instance with merged config; `api_key` re-injected. |

### 3.6 Audio Catalog (`_audio_catalog.py`)

Dedicated internal module that supplies the shared audio model catalog state consumed by both `tts.py` and `stt.py`. Factoring it out avoids duplicating module-level state and prevents an asymmetric import dependency between two sibling modules that need the same data.

**Exported names** (designed for explicit per-name imports so unit tests can patch individual pieces in isolation):
- `AudioModelId = str` — type alias documenting intent without imposing static restriction.
- `_FALLBACK_AUDIO_MODEL_IDS: list[str]` — static list used as initial cache value and as the silent-failure fallback. Covers both TTS and STT models: `openai-audio`, `tts-1`, `elevenlabs`, `elevenmusic`, `whisper-large-v3`, `whisper-1`, `scribe`.
- `_audio_model_ids_cache: list[str]` — mutable module-level cache, initialized from the fallback.
- `_audio_model_ids_lock: threading.Lock` — guards the one-shot load.
- `_audio_model_ids_loaded: bool` — one-shot guard flag.
- `_load_audio_model_ids(api_key, *, force)` — double-checked locking loader; calls `ModelInformation.get_available_models()["audio"]` and updates `_audio_model_ids_cache`. Import of `ModelInformation` is deferred (local import) to avoid circular dependency at module level.

### 3.7 Transport and Security (`_client.py`, `_auth.py`)

**`_client.py`**:
- `HttpConfig` (frozen dataclass, slots) — `base_url` and `timeout_s`.
- `_parse_error_response(status_code, body_text, content_type)` — free function that parses the API error envelope `{status, success, error: {code, message, requestId, timestamp, details, cause}}` into a `PollinationsAPIError`. It only attempts to parse JSON if `Content-Type` contains `application/json`.
- `PollinationsHttpClient` — encapsulates an `httpx.Client` (sync) and an `httpx.AsyncClient` (async), both configured with logging event hooks. The `_log_request` / `_log_response_sync` hooks and their async variants redact the `Authorization` header before writing to logs. The `raise_for_status(resp)` method calls `_parse_error_response` and raises the structured error. Exposes:
  - `post_json`, `apost_json` — JSON POST for chat completions and TTS.
  - `post_multipart`, `apost_multipart` — multipart/form-data POST for audio transcriptions; `Content-Type` + boundary are managed by httpx automatically (must not be set manually).
  - `get`, `aget` — parametrized GET for image generation and catalog endpoints.
  - `stream_post_json`, `astream_post_json` — streaming context managers for SSE responses.

**`_auth.py`**:
- `AuthConfig` (frozen dataclass, slots) — one `api_key: str` field. The `from_env_or_value(api_key)` factory resolves the key from the argument or `POLLINATIONS_API_KEY`; it raises a `ValueError` if neither is available.

### 3.8 Exceptions (`_errors.py`)

Class hierarchy:

```
RuntimeError
  └── PollinationsError
          └── PollinationsAPIError  (slots dataclass)
```

`PollinationsAPIError` carries: `status_code`, `message`, `body`, `error_code`, `request_id`, `timestamp`, `details`, `cause`. Derived properties: `is_client_error`, `is_server_error`, `is_auth_error`, `is_validation_error`, `is_payment_required` (verifies `status_code == 402` **or** `error_code == "PAYMENT_REQUIRED"` to cover the defensive case where the gateway forwards the code with an atypical status). The `to_dict()` method serializes all fields for structured logging.

### 3.9 SSE Parser (`_sse.py`)

Contains `SSEEvent` (frozen dataclass, slots) and `iter_sse_events_from_text(text)`, a minimalist parser that splits blocks by `\n\n` and extracts `data:` lines. It is used for processing SSE responses in full text format (not line-by-line streaming). Real-time streaming in `ChatPollinations` uses the inline helpers `_iter_sse_json_events_sync` / `_iter_sse_json_events_async` defined in `chat.py`, which consume `iter_lines()` / `aiter_lines()` directly from the `httpx` response.

## 4) Data Flow — Chat

### 4.1 Synchronous Flow (`invoke`)

```
User: llm.invoke([HumanMessage(...)])
  │
  ├─ ChatPollinations._generate()
  │    ├─ lc_messages_to_openai(messages)          # _openai_compat.py
  │    │    └─ _normalize_message_content()
  │    │         └─ _normalize_content_part()       # multimodal dispatcher
  │    │              ├─ _normalize_input_audio_part()
  │    │              ├─ _normalize_video_url_part()
  │    │              └─ _normalize_file_part()
  │    │
  │    ├─ JSON payload construction
  │    │    (messages + ChatPollinationsConfig fields)
  │    │
  │    ├─ PollinationsHttpClient.post_json()        # _client.py
  │    │    └─ raise_for_status() → PollinationsAPIError if error
  │    │
  │    ├─ _message_content_from_message_dict()      # chat.py
  │    │    (content → content_blocks → audio.transcript)
  │    │
  │    ├─ _parse_tool_calls()                       # chat.py
  │    ├─ _usage_metadata_from_usage()              # chat.py
  │    └─ _response_metadata_from_response()        # chat.py
  │
  └─ Returns AIMessage (content, tool_calls, usage_metadata, response_metadata,
                        additional_kwargs: {audio, prompt_filter_results, ...})
```

### 4.2 Streaming Flow (`stream`)

```
User: for chunk in llm.stream([...])
  │
  ├─ ChatPollinations._stream()
  │    ├─ [Same payload normalization process as in _generate]
  │    │
  │    ├─ PollinationsHttpClient.stream_post_json() → httpx stream context manager
  │    │
  │    └─ _iter_sse_json_events_sync(resp)          # chat.py (inline SSE parser)
  │         For each JSON event:
  │           ├─ _delta_content_from_delta_dict()   # content / content_blocks / audio
  │           ├─ _tool_call_chunks_from_delta()     # tool call chunks
  │           ├─ _text_from_any_content()           # for plain text chunk token
  │           └─ Yield AIMessageChunk
  │                (usage_metadata in the last chunk if include_usage_in_stream=True)
  │
  └─ Returns Iterator[AIMessageChunk]
```

## 5) Data Flow — Audio

### 5.1 TTS Flow (`TTSPollinations.generate`)

```
User: tts.generate("Hello, world!")
  │
  ├─ TTSPollinations.generate(text, *, params, **kwargs)
  │    ├─ _build_body(text, params, kwargs)
  │    │    ├─ _defaults_dict()                     # instance-level defaults
  │    │    ├─ merge: defaults → params → kwargs
  │    │    ├─ inject input=text (final precedence)
  │    │    ├─ SpeechRequest(**merged)              # Pydantic validation
  │    │    └─ SpeechRequest.to_body()             # model_dump(exclude_none=True)
  │    │
  │    ├─ PollinationsHttpClient.post_json(         # _client.py
  │    │       "/v1/audio/speech", body)
  │    │    └─ raise_for_status() → PollinationsAPIError if error
  │    │
  │    └─ response.content
  │
  └─ Returns bytes  (raw audio: mp3 / opus / aac / flac / wav / pcm)
```

### 5.2 STT Flow (`STTPollinations.transcribe`)

```
User: stt.transcribe(audio_bytes)
  │
  ├─ STTPollinations.transcribe(audio, *, params, **kwargs)
  │    ├─ _build_multipart(audio, params, kwargs)
  │    │    ├─ _defaults_dict()                     # instance-level defaults incl. file_name
  │    │    ├─ merge: defaults → params → kwargs
  │    │    ├─ pop file_name (extra="forbid" in TranscriptionParams)
  │    │    ├─ TranscriptionParams(**merged)        # Pydantic validation
  │    │    ├─ TranscriptionParams.to_form_data()  # model_dump + cast to str
  │    │    ├─ _mime_type_for(file_name)            # extension → MIME lookup
  │    │    └─ files_dict = {"file": (file_name, audio, mime_type)}
  │    │
  │    ├─ PollinationsHttpClient.post_multipart(    # _client.py
  │    │       "/v1/audio/transcriptions",
  │    │       files=files_dict, data=form_data)
  │    │    └─ raise_for_status() → PollinationsAPIError if error
  │    │
  │    └─ _parse_transcription_response(resp, response_format)
  │         ├─ Content-Type: application/json
  │         │    └─ TranscriptionResponse.model_validate(resp.json())
  │         └─ other Content-Type
  │              └─ resp.text  (plain str for text/srt/vtt)
  │
  └─ Returns TranscriptionResponse | str
```

## 6) Dynamic Model Catalogs

All three model catalogs (text, image, audio) implement the same pattern for lazy, thread-safe, one-shot remote loading:

```python
# Module variables
_xxx_model_ids_cache: list[str] = list(_FALLBACK_XXX_MODEL_IDS)
_xxx_model_ids_lock: threading.Lock = threading.Lock()
_xxx_model_ids_loaded: bool = False

def _load_xxx_model_ids(api_key, *, force=False) -> list[str]:
    # 1. Quick read without lock (common case: already loaded)
    if _xxx_model_ids_loaded and not force:
        return list(_xxx_model_ids_cache)
    # 2. Double-checked locking
    with _xxx_model_ids_lock:
        if _xxx_model_ids_loaded and not force:
            return list(_xxx_model_ids_cache)
        try:
            ids = ModelInformation(api_key=api_key).get_available_models().get("xxx", [])
            if ids:
                _xxx_model_ids_cache = ids
        except Exception:
            pass  # keeps fallback
        _xxx_model_ids_loaded = True
    return list(_xxx_model_ids_cache)
```

The loading is performed **at most once per process** during the first instantiation of `ChatPollinations`, `ImagePollinations`, `TTSPollinations`, or `STTPollinations`. The `ModelInformation` import is local (inside the function) to avoid circular imports between the respective modules and `models.py`.

**Catalog location by module:**

| Catalog | Module | Endpoint | Consumed by |
|---|---|---|---|
| Text model IDs | `chat.py` (inline) | `GET /text/models` | `ChatPollinations` |
| Image model IDs | `image.py` (inline) | `GET /image/models` | `ImagePollinations` |
| Audio model IDs | `_audio_catalog.py` (shared) | `GET /audio/models` | `TTSPollinations`, `STTPollinations` |

The audio catalog is factored into its own module because two independent sibling modules (`tts.py` and `stt.py`) share the same cache state. Inlining it in either one would create an asymmetric dependency between the two.

## 7) Multimodal Type Management

### Input Content Parts (Request)

```python
ContentBlock = (
    ContentBlockText            # {type: "text", text: str}
    | ContentBlockImageUrl      # {type: "image_url", image_url: {url, detail?, mime_type?}}
    | ContentBlockInputAudio    # {type: "input_audio", input_audio: {data: str, format: str}}
    | ContentBlockVideoUrl      # {type: "video_url", video_url: {url, mime_type?}}
    | ContentBlockFile          # {type: "file", file: {file_data?, file_id?, file_url?, ...}}
    | ContentBlockThinking      # {type: "thinking", thinking: str}
    | ContentBlockRedactedThinking  # {type: "redacted_thinking", data: str}
    | dict[str, Any]
)
```

`_normalize_content_part` removes the `id` field from all parts (rejected by the gateway), normalizes flat variants of `input_audio`, `video_url`, and `file` to their canonical forms, and rewrites the `"audio"` type as `"input_audio"` for compatibility with external SDKs.

### Response Types (Chat Response)

| TypedDict | Module | Key Fields |
|---|---|---|
| `ChatCompletionResponse` | `_openai_compat` | `id`, `object`, `created`, `model`, `choices`, `usage?`, `user_tier?`, `citations?`, `prompt_filter_results?` |
| `AudioTranscript` | `_openai_compat` | `transcript?`, `data?`, `id?`, `expires_at?` |
| `ContentFilterDetail` | `_openai_compat` | `filtered`, `severity`, `detected` |
| `ContentFilterResult` | `_openai_compat` | `hate?`, `self_harm?`, `sexual?`, `violence?`, `jailbreak?`, ... |
| `PromptFilterResultItem` | `_openai_compat` | `prompt_index`, `content_filter_results` |

`UserTier = Literal["anonymous", "spore", "seed", "flower", "nectar"]` is the type for the `user_tier` field in the root response; it is exposed in `AIMessage` `response_metadata`.

### Audio Types (TTS and STT)

| Type | Module | Notes |
|---|---|---|
| `AudioFormat` | `tts.py` | Closed `Literal` for TTS output encoding: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`. |
| `VoiceId` | `tts.py` | Open `str` alias; 34 known voices at release time. |
| `AudioInputFormat` | `stt.py` | Closed `Literal` for accepted input audio file extensions: `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`. |
| `TranscriptionFormat` | `stt.py` | Closed `Literal` for transcription output format: `json`, `text`, `srt`, `verbose_json`, `vtt`. |
| `TranscriptionResponse` | `stt.py` | Pydantic (`extra="allow"`); declares only `text: str`; verbose JSON extras land in `model_extra`. |

## 8) Tool Calling Management

The full tool calling flow passes through three points in the code:

1. **Definition** (`bind_tools` in `ChatPollinations`): each tool passes through `tool_to_openai_tool()` in `_openai_compat.py`, which tries in order: pass-through if it's an OpenAI dict, LangChain's `convert_to_openai_tool`, inference from `args_schema`/`tool_call_schema`/`get_input_schema()`/`BaseModel`/`TypedDict`. Provider built-ins (`google_search`, `code_execution`, etc.) are passed as dicts and returned unmodified.

2. **Synchronous Execution** (`_parse_tool_calls` in `chat.py`): separates response tool calls into valid (parsable JSON) and invalid (parsing error), which are included in `AIMessage.invalid_tool_calls`.

3. **Streaming Execution** (`_tool_call_chunks_from_delta`): reconstructs `ToolCallChunk` from the `index`, `id`, `function.name`, and `function.arguments` fields of the delta.

Inverse normalization in `lc_messages_to_openai` converts `AIMessage.tool_calls` + `invalid_tool_calls` back to OpenAI format using `_lc_tool_call_to_openai_tool_call`. The `name` field is omitted in `ToolMessage` (documented provider gateway incompatibility).

## 9) Threading and Async Considerations

- **Shared State is Module-Level, Not Instance-Level**: All three model catalogs are stored as module-level variables (`_xxx_model_ids_cache`, `_xxx_model_ids_loaded`, `_xxx_model_ids_lock`). Multiple instances of `ChatPollinations`, `ImagePollinations`, `TTSPollinations`, or `STTPollinations` within the same process share the same underlying cache without any instance-to-instance coordination required.
- **No Event Loop Coupling**: `PollinationsHttpClient` instantiates both `httpx.Client` and `httpx.AsyncClient` eagerly in `__init__`. Async methods simply call `await self._aclient.post(...)` etc. There is no `asyncio.get_event_loop()` call or manual loop management in the library; coroutines are always run by the caller's event loop.
- **`PrivateAttr` for httpx Clients**: All Pydantic-based classes store the HTTP client in a `PrivateAttr`. This keeps the transport object outside Pydantic's field schema, preventing serialization, validation, or repr exposure of the internal `httpx` handles.
- **Async Streaming vs. Async Non-Streaming**: `ChatPollinations` uses `stream_post_json` / `astream_post_json` (which return `httpx` stream context managers) only for the streaming path. All other endpoints (`post_json`, `post_multipart`, `get`) buffer the full response before returning, which is appropriate for their payload sizes (JSON, audio bytes, images).
- **Multipart and Thread Safety**: `post_multipart` and `apost_multipart` in `PollinationsHttpClient` use the respective sync and async clients. They are subject to the same thread-safety guarantees as all other methods: `httpx.Client` is safe for concurrent use across threads; `httpx.AsyncClient` must be used only within the event loop that created it.