```markdown
# API Reference

## Overview

The `langchain-pollinations` library provides a comprehensive LangChain-compatible interface to the Pollinations.ai API. This reference covers the main classes exposed by the library, their instantiation parameters, methods, and return types.

## Installation

```bash
pip install langchain-pollinations
```

## Authentication

All classes require authentication via an API key, which can be provided in two ways:

1. **Environment variable**: Set `POLLINATIONS_API_KEY` in your environment
2. **Constructor parameter**: Pass `api_key="your-key"` when instantiating

If no key is provided, a `ValueError` will be raised immediately.

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

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | API key for authentication. Falls back to `POLLINATIONS_API_KEY` environment variable |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | Base URL for the Pollinations API |
| `timeout_s` | `float` | `120.0` | HTTP request timeout in seconds |
| `request_defaults` | `ChatPollinationsConfig \| None` | `None` | Default configuration for all requests |
| `include_usage_in_stream` | `bool` | `True` | Whether to include token usage in streaming responses |
| `preserve_multimodal_deltas` | `bool` | `True` | Whether to preserve multimodal content in streaming chunks |

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

### ImagePollinations

`ImagePollinations` wraps the Pollinations image generation endpoint, supporting both image and video generation with extensive configuration options.

#### Instantiation

```python
from langchain_pollinations import ImagePollinations

image_gen = ImagePollinations(
    api_key="your-api-key",
    model="zimage",  # Default model
    width=1024,
    height=1024,
    seed=42,
    enhance=True
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | API key for authentication |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | Base URL for the API |
| `timeout_s` | `float` | `120.0` | HTTP timeout in seconds |
| `model` | `ImageModelId \| None` | `None` | Image/video model to use |
| `width` | `int \| None` | `None` | Output width in pixels |
| `height` | `int \| None` | `None` | Output height in pixels |
| `seed` | `int \| None` | `None` | Random seed (-1 to 2147483647) |
| `enhance` | `bool \| None` | `None` | Enable prompt enhancement |
| `negative_prompt` | `str \| None` | `None` | Negative prompt for guidance |
| `safe` | `bool \| None` | `None` | Enable content safety filter |
| `quality` | `Quality \| None` | `None` | Quality level: `"low"`, `"medium"`, `"high"`, `"hd"` |
| `image` | `str \| None` | `None` | Reference image URLs (comma or pipe separated) |
| `transparent` | `bool \| None` | `None` | Enable transparency (gptimage only) |
| `duration` | `int \| None` | `None` | Video duration in seconds (1-10, video models only) |
| `aspect_ratio` | `str \| None` | `None` | Aspect ratio (e.g., `"16:9"`) |
| `audio` | `bool \| None` | `None` | Include audio in video (veo only) |

#### Available Models

`ImageModelId` includes: `"kontext"`, `"nanobanana"`, `"nanobanana-pro"`, `"seedream"`, `"seedream-pro"`, `"gptimage"`, `"gptimage-large"`, `"flux"`, `"zimage"`, `"veo"`, `"seedance"`, `"seedance-pro"`, `"wan"`, `"klein"`, `"klein-large"`, `"imagen-4"`, `"grok-video"`, `"ltx-2"`

#### Core Methods

##### `generate(prompt, params=None, **kwargs)`

Synchronously generate an image or video.

**Parameters:**
- `prompt` (`str`): Text description of the desired output
- `params` (`dict[str, Any] | None`): Additional query parameters
- `**kwargs`: Parameter overrides

**Returns:** `bytes` containing the image or video data

**Example:**
```python
image_bytes = image_gen.generate("A serene mountain landscape at sunset")
with open("output.png", "wb") as f:
    f.write(image_bytes)
```

##### `agenerate(prompt, params=None, **kwargs)`

Asynchronously generate an image or video.

**Returns:** `bytes`

##### `invoke(input, config=None, **kwargs)`

LangChain-compatible synchronous generation.

**Parameters:**
- `input` (`str`): Prompt text
- `config`: LangChain config (unused)
- `**kwargs`: Additional parameters

**Returns:** `bytes`

##### `ainvoke(input, config=None, **kwargs)`

LangChain-compatible asynchronous generation.

**Returns:** `bytes`

##### `with_params(**overrides)`

Create a new instance with updated default parameters.

**Parameters:**
- `**overrides`: Parameters to override

**Returns:** New `ImagePollinations` instance

**Example:**
```python
pixel_art = image_gen.with_params(model="nanobanana", width=512, height=512)
result = pixel_art.generate("A pixel art castle")
```

##### `generate_response(prompt, params=None, **kwargs)`

Low-level method returning the raw HTTP response.

**Returns:** `httpx.Response`

#### API Endpoint Mapping

- **Endpoint**: `GET /image/{prompt}`
- **Features**: Text-to-image and text-to-video generation with 18 different models, support for reference images, quality control, and video-specific options

---

### AccountInformation

`AccountInformation` provides access to account-related endpoints including profile, balance, API key management, and usage statistics.

#### Instantiation

```python
from langchain_pollinations import AccountInformation

account = AccountInformation(
    api_key="your-api-key",
    base_url="https://gen.pollinations.ai",
    timeout_s=120.0
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | API key for authentication |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | Base URL |
| `timeout_s` | `float` | `120.0` | HTTP timeout |

#### Core Methods

##### `get_profile()`

Retrieve account profile information.

**Returns:** `dict[str, Any]` with profile details (email, tier, etc.)

**Example:**
```python
profile = account.get_profile()
print(profile["email"])
```

##### `aget_profile()`

Async version of `get_profile()`.

##### `get_balance()`

Check current account balance.

**Returns:** `dict[str, Any]` with balance and currency information

**Example:**
```python
balance = account.get_balance()
print(f"Balance: {balance['amount']} {balance['currency']}")
```

##### `aget_balance()`

Async version of `get_balance()`.

##### `get_key()`

Retrieve API key metadata.

**Returns:** `dict[str, Any]` with key information

##### `aget_key()`

Async version of `get_key()`.

##### `get_usage(params=None)`

Fetch detailed usage logs.

**Parameters:**
- `params` (`AccountUsageParams | None`): Filtering options
  - `format` (`"json" | "csv"`): Output format (default: `"json"`)
  - `limit` (`int`): Maximum records (1-50000, default: 100)
  - `before` (`str | None`): ISO timestamp for pagination

**Returns:** `dict[str, Any]` (JSON) or `str` (CSV)

**Example:**
```python
from langchain_pollinations.account import AccountUsageParams

usage = account.get_usage(
    AccountUsageParams(format="json", limit=50)
)
```

##### `aget_usage(params=None)`

Async version of `get_usage()`.

##### `get_usage_daily(params=None)`

Fetch daily aggregated usage statistics.

**Parameters:**
- `params` (`AccountUsageDailyParams | None`):
  - `format` (`"json" | "csv"`): Output format

**Returns:** `dict[str, Any]` or `str`

##### `aget_usage_daily(params=None)`

Async version of `get_usage_daily()`.

#### API Endpoint Mapping

- `GET /account/profile` → Profile details
- `GET /account/balance` → Current balance
- `GET /account/key` → API key information
- `GET /account/usage` → Detailed usage logs
- `GET /account/usage/daily` → Daily usage aggregates

---

### ModelInformation

`ModelInformation` provides discovery of available text and image models via the Pollinations API.

#### Instantiation

```python
from langchain_pollinations import ModelInformation

models = ModelInformation(
    api_key="your-api-key",
    base_url="https://gen.pollinations.ai",
    timeout_s=120.0
)
```

#### Constructor Parameters

Same as `AccountInformation`.

#### Core Methods

##### `get_available_models()`

Retrieve all available model IDs categorized by type.

**Returns:** `dict[str, list[str]]` with keys `"text"` and `"image"`

**Example:**
```python
models_dict = models.get_available_models()
print("Text models:", models_dict["text"])
print("Image models:", models_dict["image"])
```

##### `aget_available_models()`

Async version of `get_available_models()`.

##### `list_compatible_models()`

List OpenAI-compatible models.

**Returns:** `dict[str, Any]` with full model details

##### `alist_compatible_models()`

Async version.

##### `list_text_models()`

Fetch detailed text model information.

**Returns:** `list[dict[str, Any]] | dict[str, Any]`

##### `alist_text_models()`

Async version.

##### `list_image_models()`

Fetch detailed image model information.

**Returns:** `list[dict[str, Any]] | dict[str, Any]`

##### `alist_image_models()`

Async version.

#### API Endpoint Mapping

- `GET /v1/models` → OpenAI-compatible models
- `GET /text/models` → Text model details
- `GET /image/models` → Image model details

---

### PollinationsAPIError

`PollinationsAPIError` is the primary exception class raised by the library when API requests fail. It provides structured access to error details.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `status_code` | `int` | HTTP status code (e.g., 400, 401, 500) |
| `message` | `str` | Human-readable error message |
| `body` | `str \| None` | Raw response body |
| `error_code` | `str \| None` | API error code (e.g., `"BAD_REQUEST"`, `"UNAUTHORIZED"`) |
| `request_id` | `str \| None` | Unique request identifier for debugging |
| `timestamp` | `str \| None` | ISO timestamp of the error |
| `details` | `dict[str, Any] \| None` | Additional error details |
| `cause` | `Any \| None` | Root cause information |

#### Properties

##### `is_client_error`

**Returns:** `bool` - `True` if status code is 4xx

##### `is_server_error`

**Returns:** `bool` - `True` if status code is 5xx

##### `is_auth_error`

**Returns:** `bool` - `True` if status code is 401 or 403

##### `is_validation_error`

**Returns:** `bool` - `True` if status is 400 with error code `"BAD_REQUEST"`

#### Methods

##### `to_dict()`

Convert the error to a dictionary for logging.

**Returns:** `dict[str, Any]`

#### Usage Example

```python
from langchain_pollinations import ChatPollinations, PollinationsAPIError

chat = ChatPollinations(api_key="invalid-key")

try:
    response = chat.invoke([HumanMessage(content="Hello")])
except PollinationsAPIError as e:
    print(f"Error {e.status_code}: {e.message}")
    if e.is_auth_error:
        print("Authentication failed!")
    if e.request_id:
        print(f"Request ID: {e.request_id}")
```

---

## Advanced Usage

### Streaming with Usage Tracking

```python
chat = ChatPollinations(
    model="openai",
    include_usage_in_stream=True
)

for chunk in chat.stream([HumanMessage(content="Explain quantum computing")]):
    print(chunk.content, end="", flush=True)
    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
        print(f"\nTokens used: {chunk.usage_metadata['total_tokens']}")
```

### Structured Output

```python
chat = ChatPollinations(
    model="openai",
    response_format={"type": "json_object"}
)

response = chat.invoke([
    HumanMessage(content="List 3 colors in JSON format with 'colors' array")
])
print(response.content)  # JSON string
```

### Built-in Tools

```python
chat = ChatPollinations(
    model="gemini-fast",
    tools=[{"type": "google_search"}]
)

response = chat.invoke([
    HumanMessage(content="Search for the latest AI news and summarize")
])
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `POLLINATIONS_API_KEY` | Default API key for authentication |
| `POLLINATIONS_HTTP_DEBUG` | Set to `1`, `true`, `yes`, or `on` to enable HTTP debug logging |

---

## Error Handling Best Practices

```python
try:
    response = chat.invoke(message)
except PollinationsAPIError as e:
    if e.is_auth_error:
        # Handle authentication errors
        pass
    elif e.is_validation_error:
        # Handle invalid parameters
        print(f"Invalid request: {e.details}")
    elif e.is_server_error:
        # Handle server errors (retry logic)
        pass
    else:
        # Handle other errors
        print(e.to_dict())
```

---

## Support

For issues, feature requests, or questions, please refer to the official Pollinations.ai documentation or the library's repository.
