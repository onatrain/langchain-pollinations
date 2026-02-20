<div align="center">
    <table>
        <tr>
            <td width="128px">
                <img src="https://i.ibb.co/9mSDhX9Y/doki.png" alt="langchain-pollinations" width="128px"/>
            </td>
            <td align="left">
                <h1>langchain-pollinations</h1>
                <p><strong>A LangChain compatible provider library for Pollinations.ai</strong></p>
            </td>
        </tr>
    </table>

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/onatrain/langchain-pollinations)
[![Coverage](https://img.shields.io/badge/coverage-97%25-9C27B0)](https://github.com/onatrain/langchain-pollinations)
[![Status](https://img.shields.io/badge/status-beta-FFDF76)](https://github.com/onatrain/langchain-pollinations)
[![Version](https://img.shields.io/badge/version-v0.2.5b1-blue)](https://github.com/onatrain/langchain-pollinations)
[![License](https://img.shields.io/badge/license-MIT-97CA00)](https://opensource.org/license/MIT)
[![Python Versions](https://img.shields.io/badge/python-3.11+-3776AB?logo=python)](https://github.com/onatrain/langchain-pollinations)
<br>
[![LangChain](https://img.shields.io/badge/langchain-1d3d3c?logo=langchain)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/langgraph-053d5b?logo=langgraph)](https://www.langchain.com/langgraph)
</div>

---

**langchain-pollinations** provides LangChain-native wrappers for the [Pollinations.ai](https://enter.pollinations.ai) API, designed to plug into the modern LangChain ecosystem (v1.2x) while staying strictly aligned with [Pollinations.ai endpoints](https://enter.pollinations.ai/api/docs).

The library exposes four public entry points:

- `ChatPollinations` — chat model wrapper for the OpenAI-compatible `POST /v1/chat/completions` endpoint.
- `ImagePollinations` — image and video generation wrapper for `GET /image/{prompt}`.
- `ModelInformation` — utility for listing available text, image, and OpenAI-compatible models.
- `AccountInformation` — client for querying profile, balance, API key, and usage statistics.

## Why Pollinations

[Pollinations.ai](https://enter.pollinations.ai) provides a unified gateway for text generation, vision, tool use, and multimodal media—including images, video, and audio—behind a single OpenAI-compatible API surface. This library makes that gateway usable with idiomatic LangChain patterns (`invoke`, `stream`, `bind_tools`, `with_structured_output`) while keeping the public interface minimal and all configuration strictly typed via Pydantic.

## Installation

```bash
pip install langchain-pollinations
```

## Authentication

Copy `.env.example` to `.env` and set your key:

```
POLLINATIONS_API_KEY=sk-...your_key...
```

All four main classes also accept an explicit `api_key=` parameter on construction.

## ChatPollinations

`ChatPollinations` inherits from LangChain's `BaseChatModel` and supports `invoke`, `stream`, `batch`, `ainvoke`, `astream`, `abatch`, tool calling, structured output, and multimodal messages.

### Available text models

| Group | Models |
|---|---|
| OpenAI | `openai`, `openai-fast`, `openai-large`, `openai-audio` |
| Google | `gemini`, `gemini-fast`, `gemini-large`, `gemini-legacy`, `gemini-search` |
| Anthropic | `claude`, `claude-fast`, `claude-large`, `claude-legacy` |
| Reasoning | `perplexity-reasoning`, `perplexity-fast`, `deepseek` |
| Other | `mistral`, `grok`, `kimi`, `qwen-coder`, `qwen-character`, `glm`, `minimax`, `nova-fast`, `midijourney`, `chickytutor`, `nomnom` |

### Basic chat completion

```python
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

llm = ChatPollinations(model="openai")
res = llm.invoke([HumanMessage(content="Write a short haiku about distributed systems.")])
print(res.content)
```

### Streaming

```python
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

llm = ChatPollinations(model="claude")
for chunk in llm.stream([HumanMessage(content="Explain LangGraph in three sentences.")]):
    print(chunk.content, end="", flush=True)
```

### Vision (image URL input)

```python
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

llm = ChatPollinations(model="openai")
msg = HumanMessage(content=[
    {"type": "text", "text": "Describe the image in one sentence."},
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
])
res = llm.invoke([msg])
print(res.content)
```

### Audio generation

```python
import base64
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

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
    print("Saved output.mp3 | transcript:", audio_data.get("transcript"))
```

### Thinking / Reasoning models

Enable internal reasoning with `thinking` parameter:

```python
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

llm = ChatPollinations(
    model="deepseek",
    thinking={"type": "enabled", "budget_tokens": 8000},
)
res = llm.invoke([HumanMessage(content="Prove that sqrt(2) is irrational.")])
print(res.content)
```

Or use `reasoning_effort` for models that support it:

```python
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

llm = ChatPollinations(
    model="perplexity-reasoning",
    thinking={"type": "enabled", "budget_tokens": 8000},
    reasoning_effort="high"
)
res = llm.invoke([HumanMessage(content="Prove that sqrt(2) is irrational.")])
print(res.content)
```

### Tool calling
```python
import dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It is sunny in {city}."

llm = ChatPollinations(model="openai")

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

res = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]},
)

for msg in res["messages"]:
    print(f"{msg.type}: {msg.content}")
    print("*" * 100)
```

### Tool binding

```python
import dotenv, pprint
from langchain_pollinations import ChatPollinations
from langchain_core.tools import tool

dotenv.load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It is sunny in {city}."

llm = ChatPollinations(model="openai").bind_tools([get_weather])
res = llm.invoke("What is the weather in Caracas?")

print("Response type:", type(res), "\n")
pprint.pprint(res.model_dump())

print("\nTool call:")
pprint.pprint(res.tool_calls)
```

`bind_tools` also accepts Pollinations built-in tools by type string:

```python
llm = ChatPollinations(model="gemini").bind_tools([
    {"type": "google_search"},
    {"type": "code_execution"},
])
```

### Structured output

```python
import dotenv
from pydantic import BaseModel
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

llm = ChatPollinations(model="openai").with_structured_output(MovieReview)
review = llm.invoke("Review the movie Interstellar.")
print(review)
```

### Async usage

All blocking methods have async counterparts: `ainvoke`, `astream`, `abatch`.

```python
import asyncio
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

async def main():
    llm = ChatPollinations(model="gemini-fast")
    async for chunk in llm.astream([HumanMessage(content="List 3 Python tips.")]):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

## ImagePollinations

`ImagePollinations` targets `GET /image/{prompt}` and supports synchronous and asynchronous generation of images and videos with full LangChain `invoke`/`ainvoke` compatibility.

### Available image / video models

| Type | Models |
|---|---|
| Image | `flux`, `zimage`, `klein`, `klein-large`, `nanobanana`, `nanobanana-pro`, `seedream`, `seedream-pro`, `kontext` |
| Image (quality) | `gptimage`, `gptimage-large`, `imagen-4` |
| Video | `veo`, `grok-video`, `seedance`, `seedance-pro`, `wan`, `ltx-2` |

### Basic image generation

```python
import dotenv
from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

img = ImagePollinations(model="flux", width=1024, height=1024, seed=42)
data = img.generate("a cyberpunk city at night, neon lights")
with open("city.jpg", "wb") as f:
    f.write(data)
```

### Fluent interface with `with_params()`

`with_params()` returns a new pre-configured instance without mutating the original, making it easy to create specialized generators from a shared base:

```python
import dotenv
from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

base = ImagePollinations(model="flux", width=1024, height=1024)

pixel_art = base.with_params(model="klein", enhance=True)
portrait  = base.with_params(width=768, height=1024, safe=True)

data1 = pixel_art.generate("a pixel art knight standing on a cliff")
with open("knight.jpg", "wb") as f:
    f.write(data1)
data2 = portrait.generate("a watercolor portrait of a scientist")
with open("scientist.jpg", "wb") as f:
    f.write(data2)
```

### Video generation

```python
import dotenv
from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

vid = ImagePollinations(
    model="seedance", 
    duration=4, 
    aspect_ratio="16:9", 
    audio=True
)
resp = vid.generate_response("two medieval horse-knights fighting with spades at sunset, cinematic")

content_type = resp.headers.get("content-type", "")
ext = ".mp4" if "video" in content_type else ".bin"
with open(f"fighting_knights{ext}", "wb") as f:
    f.write(resp.content)
print(f"Saved fighting_knights{ext} ({len(resp.content)} bytes)")
```

### Async generation

```python
import asyncio
import dotenv
from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

async def main():
    img = ImagePollinations(model="flux")
    data = await img.agenerate("a misty forest at dawn, soft light")
    with open("forest.jpg", "wb") as f:
        f.write(data)

asyncio.run(main())
```

## ModelInformation

```python
import dotenv
from langchain_pollinations import ModelInformation

dotenv.load_dotenv()

info = ModelInformation()

# Text models
for m in info.list_text_models():
    print(
        m.get("name"), 
        "- input_modalities: ", m.get("input_modalities"),
        "- output_modalities: ", m.get("output_modalities"),
        "- tools: ", m.get("tools"),
    )
print()

# Image models
for m in info.list_image_models():
    print(
        m.get("name"), 
        "- input_modalities: ", m.get("input_modalities"),
        "- output_modalities: ", m.get("output_modalities"),
        "- tools: ", m.get("tools"),
    )
print()

# All model IDs at once
available = info.get_available_models()
print("Text models:", available["text"], "\n")
print("Image models:", available["image"], "\n")

# OpenAI-compatible /v1/models
compat = info.list_compatible_models()
print(compat)
```

Async equivalents: `alist_text_models`, `alist_image_models`, `alist_compatible_models`, `aget_available_models`.

## AccountInformation

```python
import dotenv
from langchain_pollinations import AccountInformation
from langchain_pollinations.account import AccountUsageDailyParams, AccountUsageParams

dotenv.load_dotenv()

account = AccountInformation()

balance = account.get_balance()
print(f"Balance: {balance['balance']} credits")

# Retrieve API key metadata
key_info = account.get_key()
print(key_info, "\n")

# Paginated usage logs
usage = account.get_usage(params=AccountUsageParams(limit=50, format="json"))
print(usage, "\n")

# Daily aggregated usage
daily = account.get_usage_daily(params=AccountUsageDailyParams(format="json"))
print(daily, "\n")
```

Async equivalents: `aget_profile`, `aget_balance`, `aget_key`, `aget_usage`, `aget_usage_daily`.

## Error handling

All errors surface as `PollinationsAPIError`, which carries structured fields parsed directly from the API error envelope:

```python
from langchain_pollinations import ChatPollinations, PollinationsAPIError
from langchain_core.messages import HumanMessage

try:
    llm = ChatPollinations(model="gemini", api_key="anyway")
    res = llm.invoke([HumanMessage(content="Hello")])
    print(res.content)
except PollinationsAPIError as e:
    if e.is_auth_error:
        print("Check your POLLINATIONS_API_KEY.")
    elif e.is_validation_error:
        print(f"Bad request: {e.details}")
    elif e.is_server_error:
        print(f"Server error {e.status_code} – consider retrying.")
    else:
        print(e.to_dict())
```

`PollinationsAPIError` exposes: `status_code`, `message`, `error_code`, `request_id`, `timestamp`, `details`, `cause`, and convenience properties `is_auth_error`, `is_validation_error`, `is_client_error`, `is_server_error`.

## Debug logging

Set `POLLINATIONS_HTTP_DEBUG=true` to log every outgoing request and incoming response. `Authorization` headers are automatically redacted in all log output.

```bash
POLLINATIONS_HTTP_DEBUG=true python my_script.py
```

## Contributing

Issues and pull requests are welcome—especially around edge-case compatibility with LangChain agent and tool flows, LangGraph integration, and improved ergonomics for saving generated media.

## License

Released under the [MIT License](LICENSE.md).