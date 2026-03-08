## Classes

### TTSPollinations

`TTSPollinations` is a Pydantic `BaseModel` that also implements the LangChain `Runnable[str, bytes]` interface. It wraps the Pollinations `POST /v1/audio/speech` endpoint for text-to-speech generation with dynamic audio model catalog validation and instance-level parameter defaults.

#### Instantiation

```python
from langchain_pollinations import TTSPollinations

tts = TTSPollinations(
    api_key="your-key",       # Optional if POLLINATIONS_API_KEY is set
    voice="coral",
    response_format="mp3",
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str \| None` | `None` | Optional Runnable identifier for LangChain logs and traces. |
| `api_key` | `str \| None` | `None` | API key. Falls back to `POLLINATIONS_API_KEY` environment variable. Excluded from `model_dump`. |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | API base URL. |
| `timeout_s` | `float` | `120.0` | HTTP timeout in seconds. |
| `model` | `str \| None` | `None` | Audio model ID. `None` defers to the API default. A `UserWarning` is emitted for unknown IDs. |
| `voice` | `VoiceId \| None` | `None` | Voice identifier. `None` defers to the API default (`"alloy"`). |
| `response_format` | `AudioFormat \| None` | `None` | Output audio encoding. `None` defers to the API default (`"mp3"`). |
| `speed` | `float \| None` | `None` | Playback speed multiplier, `0.25`–`4.0`. `None` defers to `1.0`. |
| `duration` | `float \| None` | `None` | Music duration in seconds, `3.0`–`300.0`. `elevenmusic` model only. |
| `instrumental` | `bool \| None` | `None` | When `True`, guarantees no vocals in the output. `elevenmusic` model only. |

All `None` instance fields are excluded from the request body; `SpeechRequest` applies its own defaults.

#### Available Audio Models

`VoiceId = str`. The audio model catalog is fetched automatically from `GET /audio/models` on the first instantiation. An unknown model ID triggers a `UserWarning` but does not block the request. To force a catalog refresh at runtime:

```python
from langchain_pollinations._audio_catalog import _load_audio_model_ids

_load_audio_model_ids(force=True)
```

Known voices at release time (34):
`alloy`, `echo`, `fable`, `onyx`, `shimmer`, `ash`, `ballad`, `coral`, `sage`, `verse`, `rachel`, `domi`, `bella`, `elli`, `charlotte`, `dorothy`, `sarah`, `emily`, `lily`, `matilda`, `adam`, `antoni`, `arnold`, `josh`, `sam`, `daniel`, `charlie`, `james`, `fin`, `callum`, `liam`, `george`, `brian`, `bill`.

#### Methods

##### `generate(text, *, params=None, **kwargs) → bytes`

Synchronously generate audio and return the raw binary content.

**Parameters:**
- `text` (`str`): The text to synthesise.
- `params` (`dict[str, Any] | None`): Optional per-call parameter overrides.
- `**kwargs`: Additional per-call overrides as keyword arguments.

**Returns:** `bytes` — binary audio in the format specified by `response_format`.

**Raises:** `PollinationsAPIError` on any non-2xx HTTP response.

```python
audio = tts.generate("Hello, world!")
with open("output.mp3", "wb") as f:
    f.write(audio)
```

##### `agenerate(text, *, params=None, **kwargs) → bytes`

Async version of `generate`.

**Raises:** `PollinationsAPIError` on any non-2xx HTTP response.

```python
audio = await tts.agenerate("Hello, world!")
```

##### `generate_response(text, *, params=None, **kwargs) → httpx.Response`

Synchronously execute the request and return the raw `httpx.Response`. Useful for inspecting headers (e.g. `Content-Type`) before consuming the binary body.

**Raises:** `PollinationsAPIError` on any non-2xx HTTP response.

##### `agenerate_response(text, *, params=None, **kwargs) → httpx.Response`

Async version of `generate_response`.

**Raises:** `PollinationsAPIError` on any non-2xx HTTP response.

##### `invoke(input, config=None, **kwargs) → bytes`

LangChain-compatible synchronous invocation. `config` is accepted for interface compatibility but is unused.

**Parameters:**
- `input` (`str`): The text to synthesise.
- `config` (`Any | None`): Unused.
- `**kwargs`: Forwarded to `generate`.

**Returns:** `bytes`

##### `ainvoke(input, config=None, **kwargs) → bytes`

LangChain-compatible asynchronous invocation.

**Returns:** `bytes`

##### `with_params(**overrides) → TTSPollinations`

Return a new `TTSPollinations` instance with merged configuration. The original instance is not mutated. The `api_key` is explicitly preserved in the copy.

```python
base  = TTSPollinations(voice="alloy", response_format="mp3")
coral = base.with_params(voice="coral")
opus  = base.with_params(response_format="opus", speed=1.25)
```

---

## SpeechRequest

Pydantic model (`extra="forbid"`) that represents the `CreateSpeechRequest` schema for `POST /v1/audio/speech`. Instantiated automatically by `_build_body`. Can be used standalone.

```python
from langchain_pollinations.tts import SpeechRequest

req = SpeechRequest(input="Hello!", voice="coral", response_format="mp3")
body = req.to_body()
```

#### Fields

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `input` | `str` | — (required) | `1`–`4096` characters | Text to synthesise. |
| `model` | `str \| None` | `None` | — | Audio model ID. `None` omitted from body. |
| `voice` | `VoiceId` | `"alloy"` | — | Voice identifier. |
| `response_format` | `AudioFormat` | `"mp3"` | — | Output audio encoding. |
| `speed` | `float` | `1.0` | `[0.25, 4.0]` | Playback speed multiplier. |
| `duration` | `float \| None` | `None` | `[3.0, 300.0]` | Music duration in seconds. `elevenmusic` only. `None` omitted from body. |
| `instrumental` | `bool \| None` | `None` | — | No-vocals flag. `elevenmusic` only. `None` omitted from body. |

#### `to_body() → dict[str, Any]`

Serialise the model to a JSON-ready dictionary. `None` fields are excluded so that model-specific parameters are not transmitted when not applicable.

---

## Type Aliases

Defined in `langchain_pollinations.tts`:

```python
AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
VoiceId     = str   # typed as str to tolerate future catalog additions
```

---

## API Endpoint

`POST /v1/audio/speech` — JSON body constructed from `SpeechRequest`. Returns binary audio in the requested `response_format`.

---

## Usage Examples

### Basic text-to-speech

```python
import dotenv
from langchain_pollinations import TTSPollinations

dotenv.load_dotenv()

tts = TTSPollinations(voice="coral", response_format="mp3")
audio = tts.generate("Welcome to Pollinations AI.")
with open("welcome.mp3", "wb") as f:
    f.write(audio)
```

### Async generation

```python
audio = await tts.agenerate("This is an async TTS call.")
```

### Different voice and format

```python
tts = TTSPollinations(voice="onyx", response_format="opus")
audio = tts.generate("Deep voice, Opus encoding.")
```

### Speed control

```python
tts = TTSPollinations(voice="alloy", speed=1.5)
audio = tts.generate("This will be spoken faster.")
```

### Per-call parameter overrides

```python
tts = TTSPollinations(voice="alloy")
audio = tts.generate(
    "Override voice and format for this call only.",
    params={"voice": "sage", "response_format": "wav"},
)
```

### Music generation with elevenmusic

```python
music = TTSPollinations(
    model="elevenmusic",
    duration=30.0,
    instrumental=True,
    response_format="mp3",
)
audio = music.generate("An upbeat jazz theme with piano and drums.")
with open("theme.mp3", "wb") as f:
    f.write(audio)
```

### Inspect raw HTTP response

```python
response = tts.generate_response("Check the Content-Type header.")
print(response.headers.get("content-type"))
audio = response.content
```

### Derive variants with `with_params`

```python
base   = TTSPollinations(response_format="mp3")
en_tts = base.with_params(voice="coral")
es_tts = base.with_params(voice="matilda", speed=0.9)
```

### LangChain chain integration

```python
from langchain_core.runnables import RunnableLambda

extract_text = RunnableLambda(lambda x: x["text"])
pipeline = extract_text | TTSPollinations(voice="coral")
audio = pipeline.invoke({"text": "Hello from a LangChain pipeline."})
```

### Force audio model catalog refresh

```python
from langchain_pollinations._audio_catalog import _load_audio_model_ids

fresh_ids = _load_audio_model_ids(force=True)
print(fresh_ids)
```

### Standalone SpeechRequest

```python
from langchain_pollinations.tts import SpeechRequest

req = SpeechRequest(
    input="Validate parameters before sending.",
    voice="echo",
    response_format="flac",
    speed=1.1,
)
print(req.to_body())
```
