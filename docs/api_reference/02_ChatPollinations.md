## Classes

### ChatPollinations

`ChatPollinations` is a LangChain-compatible chat model that wraps the Pollinations `/v1/chat/completions` endpoint. It inherits from `BaseChatModel` and supports multimodal inputs, streaming, tool calling, structured output, and advanced features like reasoning/thinking modes.

#### Instantiation

```python
from langchain_pollinations import ChatPollinations

chat = ChatPollinations(
    api_key="your-api-key",         # Optional if POLLINATIONS_API_KEY is set
    base_url="https://gen.pollinations.ai",  # Default
    timeout_s=120.0,                # Default
    model="openai",                 # ChatPollinationsConfig field as loose kwarg
    temperature=0.7,
    max_tokens=1000,
)
```

`ChatPollinationsConfig` fields can be passed directly as keyword arguments or grouped in `request_defaults=ChatPollinationsConfig(...)`. Mixing both raises `ValueError`.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | API key. Falls back to `POLLINATIONS_API_KEY` environment variable. |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | Base URL for the Pollinations API. |
| `timeout_s` | `float` | `120.0` | HTTP request timeout in seconds. |
| `request_defaults` | `ChatPollinationsConfig \| None` | `None` | Default configuration for all requests. |
| `include_usage_in_stream` | `bool` | `True` | Request token usage statistics in streaming responses. |
| `preserve_multimodal_deltas` | `bool` | `True` | Preserve structured multimodal data in streaming chunks. |
| `**kwargs` | — | — | Any `ChatPollinationsConfig` field passed directly (e.g. `model`, `temperature`). |

#### Available Text Models

`TextModelId = str`. Known values at release time (fallback catalog):

`openai`, `openai-fast`, `openai-large`, `qwen-coder`, `mistral`, `openai-audio`, `gemini`, `gemini-fast`, `deepseek`, `grok`, `gemini-search`, `chickytutor`, `midijourney`, `claude-fast`, `claude`, `claude-large`, `claude-legacy`, `perplexity-fast`, `perplexity-reasoning`, `kimi`, `gemini-large`, `gemini-legacy`, `nova-fast`, `glm`, `minimax`, `qwen-safety`.

The catalog is refreshed automatically from `GET /text/models` on the first instantiation. To force a refresh at runtime:

```python
from langchain_pollinations.chat import _load_text_model_ids

_load_text_model_ids(force=True)
```

An unknown model ID triggers a `UserWarning` but does not block the request.

#### Core Methods

##### `invoke(messages, **kwargs) → AIMessage`

Synchronously generate a single response.

**Parameters:**
- `messages` (`list[BaseMessage]`): Conversation history as LangChain messages.
- `**kwargs`: Additional configuration overrides.

**Returns:** `AIMessage` with the model's response.

```python
from langchain_core.messages import HumanMessage

response = chat.invoke([HumanMessage(content="Hello!")])
print(response.content)
```

##### `ainvoke(messages, **kwargs) → AIMessage`

Asynchronously generate a single response.

**Parameters:** Same as `invoke`.

```python
response = await chat.ainvoke([HumanMessage(content="Hello!")])
```

##### `stream(messages, **kwargs) → Iterator[AIMessageChunk]`

Synchronously stream response chunks.

```python
for chunk in chat.stream([HumanMessage(content="Tell me a story")]):
    print(chunk.content, end="", flush=True)
```

##### `astream(messages, **kwargs) → AsyncIterator[AIMessageChunk]`

Asynchronously stream response chunks.

```python
async for chunk in chat.astream([HumanMessage(content="Hello")]):
    print(chunk.content, end="", flush=True)
```

##### `bind_tools(tools, *, tool_choice=None, parallel_tool_calls=None, strict=None, **kwargs) → Runnable`

Bind tools for function calling.

**Parameters:**
- `tools` (`Sequence[dict | type | Callable | BaseTool]`): Tool definitions in any supported format.
- `tool_choice` (`str | None`): Tool selection mode (`"auto"`, `"required"`, `"none"`, or a tool name string normalized to `ToolChoiceFunction`).
- `parallel_tool_calls` (`bool | None`): Whether to allow parallel tool invocations.
- `strict` (`bool | None`): Whether to enforce strict schema validation.

**Returns:** `Runnable[LanguageModelInput, AIMessage]` with tools bound.

```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny"

chat_with_tools = chat.bind_tools([get_weather])
response = chat_with_tools.invoke([HumanMessage(content="What's the weather in Paris?")])
```

Platform builtin tools are passed as plain dicts:

```python
llm = ChatPollinations(model="gemini").bind_tools([
    {"type": "google_search"},
    {"type": "code_execution"},
    {"type": "google_maps"},
    {"type": "url_context"},
])
```

##### `with_structured_output(schema) → Runnable`

Standard LangChain structured output. Accepts Pydantic `BaseModel` subclasses or `TypedDict`.

**Parameters:**
- `schema` (`type[BaseModel] | type[TypedDict]`): Target structure.

```python
from pydantic import BaseModel

class Review(BaseModel):
    title: str
    rating: int

llm_structured = ChatPollinations(model="openai").with_structured_output(Review)
review = llm_structured.invoke("Review Interstellar.")
```

#### AIMessage Response Fields

| Field / key | Type | Description |
|---|---|---|
| `content` | `str \| list` | Text response. `list` when the model returns content blocks (thinking models). |
| `tool_calls` | `list[ToolCall]` | Parsed valid tool calls. |
| `invalid_tool_calls` | `list[InvalidToolCall]` | Tool calls that failed JSON parsing. |
| `usage_metadata` | `UsageMetadata \| None` | Token counts: `input_tokens`, `output_tokens`, `total_tokens`, `input_token_details`, `output_token_details`. |
| `response_metadata["id"]` | `str` | Response ID. |
| `response_metadata["model"]` | `str` | Model ID used. |
| `response_metadata["created"]` | `int` | Unix timestamp. |
| `response_metadata["system_fingerprint"]` | `str \| None` | Backend fingerprint. |
| `response_metadata["user_tier"]` | `UserTier \| None` | Subscription tier of the caller. |
| `response_metadata["citations"]` | `list[str] \| None` | Source URLs from search-enabled models. |
| `additional_kwargs["audio"]` | `AudioTranscript \| None` | Audio output data when `modalities` includes `"audio"`. |
| `additional_kwargs["prompt_filter_results"]` | `list[PromptFilterResultItem] \| None` | Content moderation results. |

#### Multimodal Support

`ChatPollinations` accepts multipart content in any `HumanMessage`, `AIMessage`, or `SystemMessage`.

**Image input:**
```python
message = HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])
```

**Audio input (`input_audio`):**
```python
message = HumanMessage(content=[
    {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "mp3"}},
    {"type": "text", "text": "Transcribe this."},
])
```

The `"audio"` type is also accepted and auto-normalized to `"input_audio"`. Supported formats: `wav`, `mp3`, `flac`, `opus`, `pcm16`. Flat variants (`base64`, `mime_type` on root) are normalized to canonical form automatically.

**Video input:**
```python
message = HumanMessage(content=[
    {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4", "mime_type": "video/mp4"}},
    {"type": "text", "text": "Describe this video."},
])
```

A flat variant (`{"type": "video_url", "url": "...", "mime_type": "..."}`) is also accepted and auto-normalized.

**File attachment:**
```python
message = HumanMessage(content=[
    {"type": "file", "file": {"file_url": "https://example.com/doc.pdf", "mime_type": "application/pdf"}},
    {"type": "text", "text": "Summarize this document."},
])
```

Flat variant fields (`file_url`, `file_data`, `file_id`, `file_name`, `mime_type` on root) are auto-normalized to the canonical `file` sub-object. `cache_control` on the outer part is preserved.

**Audio output generation:**
```python
import base64

llm = ChatPollinations(
    model="openai-audio",
    modalities=["text", "audio"],
    audio={"voice": "coral", "format": "mp3"},
)
res = llm.invoke([HumanMessage(content="Say hello in a friendly tone.")])
audio_data = res.additional_kwargs.get("audio", {})
if audio_data.get("data"):
    with open("output.mp3", "wb") as f:
        f.write(base64.b64decode(audio_data["data"]))
    print("transcript:", audio_data.get("transcript"))
```

#### API Endpoint

- **Endpoint**: `POST /v1/chat/completions`
- **Builtin platform tools**: `google_search`, `code_execution`, `google_maps`, `url_context`, `computer_use`, `file_search`

---

## Usage Examples

### Basic completion

```python
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

llm = ChatPollinations(model="openai")
res = llm.invoke([HumanMessage(content="Write a haiku about distributed systems.")])
print(res.content)
```

### Streaming

```python
llm = ChatPollinations(model="claude")
for chunk in llm.stream([HumanMessage(content="Explain LangGraph in three sentences.")]):
    print(chunk.content, end="", flush=True)
```

### Multimodal — image URL

```python
llm = ChatPollinations(model="openai")
msg = HumanMessage(content=[
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
])
res = llm.invoke([msg])
```

### Multimodal — video URL

```python
llm = ChatPollinations(model="gemini")
msg = HumanMessage(content=[
    {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
    {"type": "text", "text": "What happens in this video?"},
])
res = llm.invoke([msg])
```

### Audio transcription via chat (input_audio)

```python
import base64

with open("audio.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

llm = ChatPollinations(model="openai")
msg = HumanMessage(content=[
    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
    {"type": "text", "text": "Transcribe this audio."},
])
res = llm.invoke([msg])
```

### Audio output generation

```python
import base64

llm = ChatPollinations(
    model="openai-audio",
    modalities=["text", "audio"],
    audio={"voice": "coral", "format": "mp3"},
)
res = llm.invoke([HumanMessage(content="Say hello in a friendly tone.")])
audio_data = res.additional_kwargs.get("audio", {})
if audio_data.get("data"):
    with open("output.mp3", "wb") as f:
        f.write(base64.b64decode(audio_data["data"]))
    print("transcript:", audio_data.get("transcript"))
```

### Thinking / reasoning

```python
llm = ChatPollinations(
    model="deepseek",
    thinking={"type": "enabled", "budget_tokens": 8000},
)
res = llm.invoke([HumanMessage(content="Prove that sqrt(2) is irrational.")])
print(res.content)
```

### Reasoning effort

```python
llm = ChatPollinations(model="perplexity-reasoning", reasoning_effort="high")
res = llm.invoke([HumanMessage(content="What are the main causes of inflation?")])
print(res.content)
```

### Tool calling

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny in {city}."

llm = ChatPollinations(model="openai").bind_tools([get_weather])
res = llm.invoke("What is the weather in Caracas?")
print(res.tool_calls)
```

### Structured output

```python
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

llm = ChatPollinations(model="openai").with_structured_output(MovieReview)
review = llm.invoke("Review the movie Interstellar.")
```

### Content moderation results

```python
res = llm.invoke([HumanMessage(content="Hello")])
filters = res.additional_kwargs.get("prompt_filter_results", [])
for item in filters:
    print(item["prompt_index"], item["content_filter_results"])
```

### Citations from search models

```python
llm = ChatPollinations(model="gemini-search")
res = llm.invoke([HumanMessage(content="Latest news in AI")])
print(res.response_metadata.get("citations"))
```

### JSON structured output via response_format

```python
llm = ChatPollinations(model="openai", response_format={"type": "json_object"})
res = llm.invoke([HumanMessage(content="Return a JSON with keys: name, age.")])
```

### Usage metadata from streaming

```python
llm = ChatPollinations(model="openai", include_usage_in_stream=True)
chunks = list(llm.stream([HumanMessage(content="Hello")]))
last = chunks[-1]
print(last.usage_metadata)
```
