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
[![PyPI Version](https://img.shields.io/pypi/v/langchain-pollinations?label=PyPI&color=0073B7&logo=pypi)](https://pypi.org/project/langchain-pollinations/)
[![License](https://img.shields.io/badge/license-MIT-97CA00)](https://opensource.org/license/MIT)
[![Python Versions](https://img.shields.io/badge/python-3.11+-3776AB?logo=python)](https://github.com/onatrain/langchain-pollinations)
<br>
[![LangChain](https://img.shields.io/badge/langchain-1d3d3c?logo=langchain)](https://www.langchain.com/)
[![Pollinations](https://img.shields.io/badge/Served%20by-POLLINATIONS-8a2be2?style=flat&logoColor=white&labelColor=6a0dad)](https://enter.pollinations.ai)
[![LangGraph](https://img.shields.io/badge/langgraph-053d5b?logo=langgraph)](https://www.langchain.com/langgraph)
</div>

---

**langchain-pollinations** provides LangChain-native wrappers for the [Pollinations.ai](https://enter.pollinations.ai) API, designed to plug into the modern LangChain ecosystem (v1.2x) while staying strictly aligned with [Pollinations.ai endpoints](https://enter.pollinations.ai/api/docs).

The library exposes six public entry points:

- `ChatPollinations` — chat model wrapper for the OpenAI-compatible `POST /v1/chat/completions` endpoint.
- `ImagePollinations` — image and video generation wrapper for `GET /image/{prompt}`.
- `TTSPollinations` — text-to-speech wrapper for `POST /v1/audio/speech`.
- `STTPollinations` — speech-to-text wrapper for `POST /v1/audio/transcriptions`.
- `ModelInformation` — utility for listing available text, image, audio, and OpenAI-compatible models.
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

All main classes also accept an explicit `api_key=` parameter on construction.

## ChatPollinations

`ChatPollinations` inherits from LangChain's `BaseChatModel` and supports `invoke`, `stream`, `ainvoke`, `astream`, tool calling, structured output, and multimodal messages.

### Available text models

| Group | Models |
|---|---|
| OpenAI | `openai`, `openai-fast`, `openai-large`, `openai-audio` |
| Google | `gemini`, `gemini-fast`, `gemini-large`, `gemini-legacy`, `gemini-search` |
| Anthropic | `claude`, `claude-fast`, `claude-large`, `claude-legacy` |
| Reasoning | `perplexity-reasoning`, `perplexity-fast`, `deepseek` |
| Other | `mistral`, `grok`, `kimi`, `qwen-coder`, `qwen-safety`, `glm`, `minimax`, `nova-fast`, `midijourney`, `chickytutor` |

### Basic usage

```python
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage, SystemMessage

dotenv.load_dotenv()

llm = ChatPollinations(model="openai", temperature=0.7)
res = llm.invoke([
    SystemMessage(content="You are a concise assistant."),
    HumanMessage(content="What is the capital of France?"),
])
print(res.content)
```

### Streaming

```python
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

llm = ChatPollinations(model="gemini-fast")
for chunk in llm.stream([HumanMessage(content="List 5 Python tips.")]):
    print(chunk.content, end="", flush=True)
```

### Multimodal input

`ChatPollinations` accepts `image_url`, `input_audio`, `video_url`, and `file` content blocks inside `HumanMessage`:

```python
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

llm = ChatPollinations(model="openai-large")
res = llm.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
    ])
])
print(res.content)
```

### Tool calling

```python
import pprint
import dotenv
from langchain_core.tools import tool
from langchain_pollinations import ChatPollinations

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
    audio=True,
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

## TTSPollinations

`TTSPollinations` wraps `POST /v1/audio/speech` and converts text to audio. It supports a dynamic audio model catalog, per-instance parameter defaults, and a LangChain Runnable-compatible interface.

### Available TTS models and voices

| Models | Notes |
|---|---|
| `tts-1` | Standard OpenAI TTS |
| `elevenlabs` | ElevenLabs voices |
| `elevenmusic` | Music generation; supports `duration` and `instrumental` |

Supported output formats: `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm`.

Available voices include: `alloy`, `echo`, `fable`, `onyx`, `shimmer`, `ash`, `ballad`, `coral`, `sage`, `verse`, `rachel`, `domi`, `bella`, `elli`, `charlotte`, `dorothy`, `sarah`, `emily`, `lily`, `matilda`, `adam`, `antoni`, `arnold`, `josh`, `sam`, `daniel`, `charlie`, `james`, `fin`, `callum`, `liam`, `george`, `brian`, `bill`.

### Basic speech generation

```python
import dotenv
from langchain_pollinations import TTSPollinations

dotenv.load_dotenv()

tts = TTSPollinations(voice="rachel", response_format="mp3")
audio_bytes = tts.generate("Hello, welcome to langchain-pollinations.")
with open("speech.mp3", "wb") as f:
    f.write(audio_bytes)
```

### Per-call parameter overrides

Instance defaults can be overridden on any individual call:

```python
import dotenv
from langchain_pollinations import TTSPollinations

dotenv.load_dotenv()

tts = TTSPollinations(model="elevenlabs", voice="onyx")

# Override voice and speed for this call only
audio = tts.generate("A slower, calm narration.", speed=0.8, voice="echo")
with open("narration.mp3", "wb") as f:
    f.write(audio)
```

### Music generation with `elevenmusic`

```python
import dotenv
from langchain_pollinations import TTSPollinations

dotenv.load_dotenv()

music = TTSPollinations(
    model="elevenmusic",
    duration=30,
    instrumental=True,
    response_format="mp3",
)
audio_bytes = music.generate("An upbeat jazz theme with piano and drums")
with open("theme.mp3", "wb") as f:
    f.write(audio_bytes)
```

### LangChain pipeline

```python
import dotenv
from langchain_core.runnables import RunnableLambda
from langchain_pollinations import ChatPollinations, TTSPollinations

dotenv.load_dotenv()

llm = ChatPollinations(model="openai-fast")
tts = TTSPollinations(voice="shimmer")

pipeline = llm | RunnableLambda(lambda msg: msg.content) | tts
audio = pipeline.invoke("Summarize the water cycle in two sentences.")
with open("summary.mp3", "wb") as f:
    f.write(audio)
```

### Async generation

```python
import asyncio
import dotenv
from langchain_pollinations import TTSPollinations

dotenv.load_dotenv()

async def main():
    tts = TTSPollinations(voice="alloy", response_format="wav")
    audio = await tts.agenerate("Async text-to-speech with Pollinations.")
    with open("output.wav", "wb") as f:
        f.write(audio)

asyncio.run(main())
```

## STTPollinations

`STTPollinations` wraps `POST /v1/audio/transcriptions` and converts audio files to text using a multipart/form-data request. It supports typed response parsing, language hints, verbose JSON output, and a LangChain Runnable-compatible interface.

### Available STT models

| Model | Notes |
|---|---|
| `whisper-large-v3` | Default; high-accuracy multilingual transcription |
| `whisper-1` | Faster, lighter Whisper variant |
| `scribe` | ElevenLabs Scribe model |

Accepted audio input formats: `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`.

Response formats: `json` (default), `text`, `srt`, `verbose_json`, `vtt`.

### Basic transcription

```python
import dotenv
from langchain_pollinations import STTPollinations

dotenv.load_dotenv()

stt = STTPollinations()
with open("speech.mp3", "rb") as fh:
    result = stt.transcribe(fh.read())
print(result.text)
```

### Language hint and custom model

```python
import dotenv
from langchain_pollinations import STTPollinations

dotenv.load_dotenv()

stt = STTPollinations(model="scribe", language="es")
with open("grabacion.wav", "rb") as fh:
    result = stt.transcribe(fh.read(), filename="grabacion.wav")
print(result.text)
```

### Verbose JSON for segment-level data

When `response_format="verbose_json"`, additional metadata (segments, words, language, duration) is available via `result.model_extra`:

```python
import dotenv
from langchain_pollinations import STTPollinations

dotenv.load_dotenv()

stt = STTPollinations(response_format="verbose_json")
with open("interview.mp3", "rb") as fh:
    result = stt.transcribe(fh.read())

print(result.text)
print("Segments:", result.model_extra.get("segments"))
print("Detected language:", result.model_extra.get("language"))
```

### Plain text and subtitle formats

When `response_format` is `"text"`, `"srt"`, or `"vtt"`, the return value is a plain `str`:

```python
import dotenv
from langchain_pollinations import STTPollinations

dotenv.load_dotenv()

stt = STTPollinations(response_format="srt")
with open("video_audio.mp3", "rb") as fh:
    subtitles = stt.transcribe(fh.read())  # returns str
with open("subtitles.srt", "w") as f:
    f.write(subtitles)
```

### LangChain pipeline

```python
import dotenv
from langchain_core.runnables import RunnableLambda
from langchain_pollinations import STTPollinations, ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

stt = STTPollinations(language="en")
llm = ChatPollinations(model="openai-fast")

pipeline = (
    stt
    | RunnableLambda(lambda r: [HumanMessage(content=f"Summarize: {r.text}")])
    | llm
)

with open("meeting.mp3", "rb") as fh:
    summary = pipeline.invoke(fh.read())
print(summary.content)
```

### Async transcription

```python
import asyncio
import dotenv
from langchain_pollinations import STTPollinations

dotenv.load_dotenv()

async def main():
    stt = STTPollinations(model="whisper-large-v3")
    with open("audio.mp3", "rb") as fh:
        audio = fh.read()
    result = await stt.atranscribe(audio)
    print(result.text)

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
    )
print()

# Audio models (TTS + STT)
for m in info.list_audio_models():
    print(m.get("name") or m.get("id"))
print()

# All model IDs at once
available = info.get_available_models()
print("Text models:",  available["text"],  "\n")
print("Image models:", available["image"], "\n")
print("Audio models:", available["audio"], "\n")

# OpenAI-compatible /v1/models
compat = info.list_compatible_models()
print(compat)
```

Async equivalents: `alist_text_models`, `alist_image_models`, `alist_audio_models`, `alist_compatible_models`, `aget_available_models`.

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

# Account profile
profile = account.get_profile()
print(profile.tier, profile.created_at, "\n")

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
    elif e.is_payment_required:
        print("Insufficient balance. Some models are for paid-only use.")
    elif e.is_validation_error:
        print(f"Bad request: {e.details}")
    elif e.is_server_error:
        print(f"Server error {e.status_code} – consider retrying.")
    else:
        print(e.to_dict())
```

`PollinationsAPIError` exposes: `status_code`, `message`, `error_code`, `request_id`, `timestamp`, `details`, `cause`, and convenience properties `is_auth_error`, `is_payment_required`, `is_validation_error`, `is_client_error`, `is_server_error`.

## Debug logging

Set `POLLINATIONS_HTTP_DEBUG=true` to log every outgoing request and incoming response. `Authorization` headers are automatically redacted in all log output.

```bash
POLLINATIONS_HTTP_DEBUG=true python my_script.py
```

## Contributing

Issues and pull requests are welcome—especially around edge-case compatibility with LangChain agent and tool flows, LangGraph integration, and improved ergonomics for saving generated media.

## License

Released under the [MIT License](LICENSE.md).
