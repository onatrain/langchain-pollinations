## Overview

`langchain-pollinations` is a Python library that provides a LangChain-compatible interface to the [Pollinations.ai](https://pollinations.ai) API. It covers the full range of generative services offered by the provider: text chat completions, image and video generation, text-to-speech synthesis, speech-to-text transcription, and account management.

---

## Installation

```bash
pip install langchain-pollinations
```

---

## Authentication

All classes require a Pollinations API key. It can be supplied in two ways:

1. **Environment variable** — set `POLLINATIONS_API_KEY` in your environment before importing.
2. **Constructor parameter** — pass `api_key="your-key"` when instantiating any class.

If neither is available a `ValueError` is raised immediately (before any network call is made).

```python
import os
os.environ["POLLINATIONS_API_KEY"] = "your-key"

# or pass it explicitly
from langchain_pollinations import ChatPollinations
llm = ChatPollinations(api_key="your-key", model="openai")
```

---

## Public API

All names below are importable directly from `langchain_pollinations`:

```python
from langchain_pollinations import (
    # Main service classes
    ChatPollinations,
    ImagePollinations,
    TTSPollinations,
    STTPollinations,
    AccountInformation,
    ModelInformation,
    # Account response models
    AccountProfile,
    AccountTier,
    AccountUsageRecord,
    AccountUsageResponse,
    # Audio response and format types
    AudioInputFormat,
    TranscriptionFormat,
    TranscriptionResponse,
    # Exceptions
    PollinationsAPIError,
)
```

---

## Capabilities

### Chat completions — `ChatPollinations`

LangChain `BaseChatModel` wrapping `POST /v1/chat/completions`. Supports:

- Synchronous and asynchronous inference (`invoke`, `ainvoke`)
- Synchronous and asynchronous streaming (`stream`, `astream`)
- Multimodal inputs: text, images, audio (`input_audio`), video, files
- Audio output generation (voice synthesis via `modalities=["text", "audio"]`)
- Tool / function calling (`bind_tools`, `with_structured_output`)
- Reasoning and thinking modes (`thinking`, `reasoning_effort`, `thinking_budget`)
- All major text models: `openai`, `mistral`, `claude`, `gemini`, `deepseek`, and more

```python
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

llm = ChatPollinations(model="openai")
response = llm.invoke([HumanMessage(content="Hello!")])
print(response.content)
```

See `02_ChatPollinations.md` and `03_ChatPollinationsConfig.md` for full parameter and method reference.

---

### Image and video generation — `ImagePollinations`

LangChain `Runnable[str, bytes]` wrapping `GET /image/{prompt}`. Supports:

- Synchronous and asynchronous generation (`generate`, `agenerate`, `invoke`, `ainvoke`)
- All image models: `flux`, `turbo`, `zimage`, `kontext`, `gptimage`, `imagen-4`, and more
- Video models: `veo`, `seedance`, `grok-video`, `wan`, `ltx-2`, and more
- Per-request parameters: `width`, `height`, `seed`, `enhance`, `quality`, `transparent`, `duration`, `aspect_ratio`, `audio`
- Immutable parameter composition via `with_params(**overrides)`

```python
from langchain_pollinations import ImagePollinations

img = ImagePollinations(model="flux", width=1024, height=1024)
image_bytes = img.generate("A sunset over the ocean")

with open("output.png", "wb") as f:
    f.write(image_bytes)
```

See `05_ImagePollinations.md` for full reference.

---

### Text-to-speech — `TTSPollinations`

LangChain `Runnable[str, bytes]` wrapping `POST /v1/audio/speech`. Supports:

- Synchronous and asynchronous synthesis (`generate`, `agenerate`, `invoke`, `ainvoke`)
- Multiple audio models: `openai-audio`, `tts-1`, `elevenlabs`, `elevenmusic`
- Output formats: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`
- Voice selection via `voice` (open `str`; 34+ known voices at release)
- Speed control (`speed`, range `0.25`–`4.0`)
- Music-specific parameters: `duration` (3–300 s) and `instrumental` (`elevenmusic` only)
- Immutable parameter composition via `with_params(**overrides)`

```python
from langchain_pollinations import TTSPollinations

tts = TTSPollinations(model="openai-audio", voice="alloy", response_format="mp3")
audio_bytes = tts.generate("Welcome to langchain-pollinations.")

with open("speech.mp3", "wb") as f:
    f.write(audio_bytes)
```

See `09_TTSPollinations.md` for full reference.

---

### Speech-to-text — `STTPollinations`

LangChain `Runnable[bytes, TranscriptionResponse | str]` wrapping `POST /v1/audio/transcriptions`. Supports:

- Synchronous and asynchronous transcription (`transcribe`, `atranscribe`, `invoke`, `ainvoke`)
- Audio models: `whisper-large-v3`, `whisper-1`, `scribe`
- Input formats: `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`
- Output formats: `json` → `TranscriptionResponse`, `text`/`srt`/`vtt`/`verbose_json`
- Language hint (`language`, ISO-639-1) and context `prompt`
- Automatic MIME type detection from the `file_name` instance parameter
- Immutable parameter composition via `with_params(**overrides)`

```python
from langchain_pollinations import STTPollinations, TranscriptionResponse

stt = STTPollinations(model="whisper-large-v3", language="en")

with open("audio.mp3", "rb") as f:
    audio_bytes = f.read()

result = stt.transcribe(audio_bytes)
if isinstance(result, TranscriptionResponse):
    print(result.text)
```

See `10_STTPollinations.md` for full reference.

---

### Account management — `AccountInformation`

Access to account metadata and usage statistics via `GET /account/*`:

- `get_profile()` / `aget_profile()` → `AccountProfile`
- `get_balance()` / `aget_balance()` → `dict`
- `get_key()` / `aget_key()` → `dict`
- `get_usage(params)` / `aget_usage(params)` → `AccountUsageResponse | str`
- `get_usage_daily(params)` / `aget_usage_daily(params)` → `dict | str`

```python
from langchain_pollinations import AccountInformation

account = AccountInformation(api_key="your-key")
profile = account.get_profile()
print(profile.tier, profile.email)
```

See `06_AccountInformation.md` for full reference.

---

### Model discovery — `ModelInformation`

Enumerate available models from the Pollinations catalog:

- `list_text_models()` / `alist_text_models()` — `GET /text/models`
- `list_image_models()` / `alist_image_models()` — `GET /image/models`
- `list_audio_models()` / `alist_audio_models()` — `GET /audio/models`
- `list_compatible_models()` / `alist_compatible_models()` — `GET /v1/models`
- `get_available_models()` / `aget_available_models()` → `dict[str, list[str]]` with `"text"`, `"image"`, and `"audio"` keys

```python
from langchain_pollinations import ModelInformation

info = ModelInformation(api_key="your-key")
models = info.get_available_models()
print(models["text"])
print(models["audio"])
```

See `07_ModelInformation.md` for full reference.

---

### Error handling — `PollinationsAPIError`

All service classes raise `PollinationsAPIError` on non-2xx API responses. The exception parses the structured JSON error envelope automatically:

```python
from langchain_pollinations import PollinationsAPIError

try:
    llm.invoke([...])
except PollinationsAPIError as e:
    print(e.status_code, e.error_code, e.message)
    if e.is_payment_required:
        print("Pollen balance exhausted.")
    elif e.is_auth_error:
        print("Check your API key.")
```

See `08_PollinationsAPIError.md` for the full field and property reference.

---

## Base URL and Timeout

All classes accept `base_url` and `timeout_s` parameters with identical defaults:

| Parameter | Default |
|---|---|
| `base_url` | `"https://gen.pollinations.ai"` |
| `timeout_s` | `120.0` |

---

## Debug Logging

Set the environment variable `POLLINATIONS_HTTP_DEBUG=1` (or `true`, `yes`, `on`) to enable detailed logging of all outgoing HTTP requests and responses. The `Authorization` header is automatically redacted in all log output.

```bash
export POLLINATIONS_HTTP_DEBUG=1
```

---

## API Surface Summary

| Class | Module | LangChain base | Modality |
|---|---|---|---|
| `ChatPollinations` | `chat` | `BaseChatModel` | Text / multimodal |
| `ImagePollinations` | `image` | `Runnable[str, bytes]` | Image / video |
| `TTSPollinations` | `tts` | `Runnable[str, bytes]` | Text → audio |
| `STTPollinations` | `stt` | `Runnable[bytes, TranscriptionResponse \| str]` | Audio → text |
| `AccountInformation` | `account` | — | Account management |
| `ModelInformation` | `models` | — | Model discovery |
| `PollinationsAPIError` | `_errors` | `RuntimeError` | Error handling |