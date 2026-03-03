## Classes

### ChatPollinations

`ChatPollinations` is a LangChain-compatible chat model that wraps the Pollinations `/v1/chat/completions` endpoint. It supports multimodal inputs, streaming, tool calling, and advanced features like reasoning/thinking modes.

#### Instantiation

```python
from langchain_pollinations import ChatPollinations

chat = ChatPollinations(
    api_key="your-api-key",  # Optional if POLLINATIONS_API_KEY is set
    base_url="https://gen.pollinations.ai",  # Default
    timeout_s=120.0,  # Default
    model="openai",  # Model to use
    temperature=0.7,  # Sampling temperature
    max_tokens=1000,  # Maximum tokens to generate
    # ... other ChatPollinationsConfig parameters
)
```

Parameters from ChatPollinationsConfig (see below) can be passed directly as keyword arguments or via request_defaults=ChatPollinationsConfig(...). Mixing both raises ValueError.

#### Constructor Parameters

| Parameter | Type | Default | Description                                                                           |
|-----------|------|---------|---------------------------------------------------------------------------------------|
| `api_key` | `str \| None` | `None` | API key for authentication. Falls back to `POLLINATIONS_API_KEY` environment variable |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | Base URL for the Pollinations API                                                     |
| `timeout_s` | `float` | `120.0` | HTTP request timeout in seconds                                                       |
| `request_defaults` | `ChatPollinationsConfig \| None` | `None` | Default configuration for all requests                                                |
| `include_usage_in_stream` | `bool` | `True` | Whether to include token usage in streaming responses                                 |
| `preserve_multimodal_deltas` | `bool` | `True` | Whether to preserve multimodal content in streaming chunks                            |
| `**kwargs` | — | — | Any ChatPollinationsConfig field passed directly (e.g. model, temperature).           |

**Note**: You can also pass any `ChatPollinationsConfig` field directly as a keyword argument (e.g., `model`, `temperature`). These will be merged into `request_defaults`.

#### Request Configuration Parameters

These parameters can be set either via `request_defaults` or directly in the constructor:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"openai"` | Model identifier (e.g., `"openai"`, `"mistral"`, `"claude-3.5-sonnet"`) |
| `temperature` | `float \| None` | `None` | Sampling temperature (0.0 to 2.0) |
| `max_tokens` | `int \| None` | `None` | Maximum tokens to generate |
| `top_p` | `float \| None` | `None` | Nucleus sampling parameter |
| `frequency_penalty` | `float \| None` | `None` | Penalize frequent tokens |
| `presence_penalty` | `float \| None` | `None` | Penalize tokens already present |
| `stop` | `list[str] \| None` | `None` | Stop sequences |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `response_format` | `dict \| None` | `None` | Structured output format (e.g., `{"type": "json_object"}`) |
| `tools` | `list[dict] \| None` | `None` | Tool definitions for function calling |
| `tool_choice` | `str \| dict \| None` | `None` | Tool selection strategy |

#### Core Methods

##### `invoke(messages, **kwargs)`

Synchronously generate a single response.

**Parameters:**
- `messages` (`list[BaseMessage]`): Conversation history as LangChain messages
- `**kwargs`: Additional configuration overrides

**Returns:** `AIMessage` with the model's response

**Example:**
```python
from langchain_core.messages import HumanMessage

response = chat.invoke([HumanMessage(content="Hello!")])
print(response.content)
```

##### `ainvoke(messages, **kwargs)`

Asynchronously generate a single response.

**Parameters:** Same as `invoke`

**Returns:** `AIMessage`

**Example:**
```python
response = await chat.ainvoke([HumanMessage(content="Hello!")])
```

##### `stream(messages, **kwargs)`

Synchronously stream response chunks.

**Parameters:** Same as `invoke`

**Returns:** Iterator of `AIMessageChunk`

**Example:**
```python
for chunk in chat.stream([HumanMessage(content="Tell me a story")]):
    print(chunk.content, end="", flush=True)
```

##### `astream(messages, **kwargs)`

Asynchronously stream response chunks.

**Returns:** AsyncIterator of `AIMessageChunk`

**Example:**
```python
async for chunk in chat.astream([HumanMessage(content="Hello")]):
    print(chunk.content, end="", flush=True)
```

##### `bind_tools(tools, tool_choice=None, parallel_tool_calls=None, strict=None, **kwargs)`

Bind tools for function calling.

**Parameters:**
- `tools` (`Sequence[dict | type | Callable | BaseTool]`): Tool definitions
- `tool_choice` (`str | None`): Tool selection mode (`"auto"`, `"required"`, `"none"`, or tool name)
- `parallel_tool_calls` (`bool | None`): Whether to allow parallel tool invocations
- `strict` (`bool | None`): Whether to enforce strict schema validation

**Returns:** `Runnable` with tools bound

**Example:**
```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny"

chat_with_tools = chat.bind_tools([get_weather])
response = chat_with_tools.invoke([HumanMessage(content="What's the weather in Paris?")])
```

Builtin platform tools are passed as plain dicts:

```python
llm = ChatPollinations(model="gemini").bind_tools([
    {"type": "google_search"},
    {"type": "code_execution"},
])
```

##### `with_structured_output(schema)`

Standard LangChain structured output. Accepts Pydantic BaseModel subclasses or TypedDict.

**Parameters:**
- `schema` (`BaseModel | TypedDict`): Structure to fill with data

**Returns:** `Runnable` with structured output

```python
from pydantic import BaseModel

class Review(BaseModel):
    title: str
    rating: int

llm_structured = ChatPollinations(model="openai").with_structured_output(Review)
review = llm_structured.invoke("Review Interstellar.")
```

#### Multimodal Support

`ChatPollinations` supports images and audio in messages:

```python
from langchain_core.messages import HumanMessage

# Image input
message = HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# Audio input
message = HumanMessage(content=[
    {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "mp3"}}
])

response = chat.invoke([message])
```

#### API Endpoint Mapping

- **Endpoint**: `POST /v1/chat/completions`
- **Features**: OpenAI-compatible chat completions with extensions for multimodal content, built-in tools (`google_search`, `code_execution`), and thinking/reasoning modes

---

## Chat Usage Examples

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

### Audio generation

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

### Audio transcription (input_audio)

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

### Thinking / reasoning

```python
llm = ChatPollinations(
    model="deepseek",
    thinking={"type": "enabled", "budget_tokens": 8000},
)
res = llm.invoke([HumanMessage(content="Prove that sqrt(2) is irrational.")])
print(res.content)
print(res.additional_kwargs["reasoning_content"])
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
