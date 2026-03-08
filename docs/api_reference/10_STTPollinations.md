## Classes

### STTPollinations

`STTPollinations` is a Pydantic `BaseModel` that also implements the LangChain `Runnable[bytes, TranscriptionResponse | str]` interface. It wraps the Pollinations `POST /v1/audio/transcriptions` endpoint for speech-to-text transcription with multipart/form-data request construction, typed response parsing, and dynamic audio model catalog validation.

#### Instantiation

```python
from langchain_pollinations import STTPollinations

stt = STTPollinations(
    api_key="your-key",     # Optional if POLLINATIONS_API_KEY is set
    model="whisper-large-v3",
    language="en",
    response_format="json",
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str \| None` | `None` | Optional Runnable identifier for LangChain logs and traces. |
| `api_key` | `str \| None` | `None` | API key. Falls back to `POLLINATIONS_API_KEY` environment variable. Excluded from `model_dump`. |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | API base URL. |
| `timeout_s` | `float` | `120.0` | HTTP timeout in seconds. |
| `model` | `str \| None` | `None` | STT model ID. `None` defers to the API default (`"whisper-large-v3"`). A `UserWarning` is emitted for unknown IDs. |
| `language` | `str \| None` | `None` | ISO-639-1 language code (e.g. `"en"`, `"es"`). `None` enables automatic language detection. |
| `prompt` | `str \| None` | `None` | Optional context text to guide transcription style or continue a prior segment. |
| `response_format` | `TranscriptionFormat \| None` | `None` | Output format. `None` defers to `"json"`. |
| `temperature` | `float \| None` | `None` | Sampling temperature `0.0`–`1.0`. `None` defers to the model default. |
| `file_name` | `str` | `"audio.mp3"` | Default filename sent with the audio in the multipart body. The extension determines the MIME type. Override per-call when the audio has a different format. |

All `None` instance fields are excluded from the request form; `TranscriptionParams` applies its own defaults.

#### Methods

##### `transcribe(audio, *, params=None, **kwargs) → TranscriptionResponse | str`

Synchronously transcribe audio bytes and return a parsed response.

The return type depends on the effective `response_format`:
- `"json"` / `"verbose_json"` → `TranscriptionResponse`. For `"verbose_json"`, additional fields (`segments`, `words`, `language`, `duration`, etc.) are accessible via `result.model_extra`.
- `"text"` / `"srt"` / `"vtt"` → plain `str`.

**Parameters:**
- `audio` (`bytes`): Raw audio bytes to transcribe.
- `params` (`dict[str, Any] | None`): Optional per-call parameter overrides. May include `file_name`.
- `**kwargs`: Additional per-call overrides as keyword arguments. May include `file_name`.

**Raises:** `PollinationsAPIError` on any non-2xx HTTP response.

```python
with open("speech.mp3", "rb") as fh:
    result = stt.transcribe(fh.read())
print(result.text)
```

##### `atranscribe(audio, *, params=None, **kwargs) → TranscriptionResponse | str`

Async version of `transcribe`.

**Raises:** `PollinationsAPIError` on any non-2xx HTTP response.

```python
result = await stt.atranscribe(audio_bytes)
print(result.text)
```

##### `transcribe_response(audio, *, params=None, **kwargs) → httpx.Response`

Synchronously execute the request and return the raw `httpx.Response`. Useful for inspecting headers or consuming the body manually.

**Raises:** `PollinationsAPIError` on any non-2xx HTTP response.

##### `atranscribe_response(audio, *, params=None, **kwargs) → httpx.Response`

Async version of `transcribe_response`.

**Raises:** `PollinationsAPIError` on any non-2xx HTTP response.

##### `invoke(input, config=None, **kwargs) → TranscriptionResponse | str`

LangChain-compatible synchronous invocation. `config` is accepted for interface compatibility but is unused.

**Parameters:**
- `input` (`bytes`): Raw audio bytes to transcribe.
- `config` (`Any | None`): Unused.
- `**kwargs`: Forwarded to `transcribe`.

**Returns:** `TranscriptionResponse | str`

##### `ainvoke(input, config=None, **kwargs) → TranscriptionResponse | str`

LangChain-compatible asynchronous invocation.

**Returns:** `TranscriptionResponse | str`

##### `with_params(**overrides) → STTPollinations`

Return a new `STTPollinations` instance with merged configuration. The original instance is not mutated. The `api_key` is explicitly preserved in the copy.

```python
base    = STTPollinations(model="whisper-large-v3")
es_stt  = base.with_params(language="es")
verbose = base.with_params(response_format="verbose_json", temperature=0.0)
```

---

## TranscriptionParams

Pydantic model (`extra="forbid"`) representing the non-file form fields for `POST /v1/audio/transcriptions`. Instantiated automatically by `_build_multipart`. Can be used standalone.

```python
from langchain_pollinations.stt import TranscriptionParams

p = TranscriptionParams(model="whisper-large-v3", language="en", response_format="json")
form = p.to_form_data()
```

#### Fields

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `model` | `str` | `"whisper-large-v3"` | — | STT model identifier. |
| `language` | `str \| None` | `None` | — | ISO-639-1 language code. `None` omitted from form. |
| `prompt` | `str \| None` | `None` | — | Context text. `None` omitted from form. |
| `response_format` | `TranscriptionFormat` | `"json"` | — | Output format. |
| `temperature` | `float \| None` | `None` | `[0.0, 1.0]` | Sampling temperature. `None` omitted from form. |

#### `to_form_data() → dict[str, str]`

Serialise the model to a string dictionary suitable for the `data=` parameter of an httpx multipart POST. `None` fields are excluded. All values are cast to `str`.

---

## TranscriptionResponse

Pydantic model (`extra="allow"`) for the JSON body returned by `POST /v1/audio/transcriptions`.

```python
class TranscriptionResponse(BaseModel):
    text: str
```

The `text` field contains the full transcription and is present for both `"json"` and `"verbose_json"` response formats. When `"verbose_json"` is requested, additional fields returned by the model (`language`, `duration`, `segments`, `words`, etc.) are preserved via `extra="allow"` and accessible through `instance.model_extra`.

---

## Type Aliases

Defined in `langchain_pollinations.stt`:

```python
TranscriptionFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]
AudioInputFormat    = Literal["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
```

`AudioInputFormat` lists the audio file formats accepted by the API endpoint.

### MIME Type Map

The module maintains an internal mapping from audio file extension (without dot, lowercase) to MIME type, used to populate the `Content-Type` of the multipart file part:

| Extension | MIME type |
|---|---|
| `mp3` | `audio/mpeg` |
| `mp4` | `audio/mp4` |
| `mpeg` | `audio/mpeg` |
| `mpga` | `audio/mpeg` |
| `m4a` | `audio/mp4` |
| `wav` | `audio/wav` |
| `webm` | `audio/webm` |

Unrecognised extensions fall back to `application/octet-stream`.

---

## API Endpoint

`POST /v1/audio/transcriptions` — multipart/form-data body with:
- `file`: `(filename, audio_bytes, mime_type)`
- form fields from `TranscriptionParams.to_form_data()`

---

## Module-level Helper

### `_load_audio_model_ids`

Imported from `langchain_pollinations._audio_catalog`. The remote call is made **at most once per process lifetime**. On failure the fallback catalog is retained without raising.

```python
from langchain_pollinations._audio_catalog import _load_audio_model_ids

fresh_ids = _load_audio_model_ids(force=True)
```

**Parameters:**
- `api_key` (`str | None`): Forwarded to `ModelInformation`. Resolved from `POLLINATIONS_API_KEY` when `None`.
- `force` (`bool`): Bypass the one-shot guard and re-fetch.

**Returns:** A copy of the current audio model ID list.

---

## Usage Examples

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

### Async transcription

```python
with open("speech.mp3", "rb") as fh:
    audio = fh.read()
result = await stt.atranscribe(audio)
print(result.text)
```

### Language hint and custom model

```python
stt = STTPollinations(model="scribe", language="es")
with open("grabacion.wav", "rb") as fh:
    result = stt.transcribe(fh.read(), file_name="grabacion.wav")
print(result.text)
```

### Verbose JSON for segment-level data

```python
stt = STTPollinations(response_format="verbose_json")
with open("speech.mp3", "rb") as fh:
    result = stt.transcribe(fh.read())
print(result.text)
print(result.model_extra.get("segments"))
print(result.model_extra.get("language"))
print(result.model_extra.get("duration"))
```

### Plain-text output format

```python
stt = STTPollinations(response_format="text")
with open("speech.mp3", "rb") as fh:
    transcript = stt.transcribe(fh.read())  # returns str
print(transcript)
```

### SRT subtitle output

```python
stt = STTPollinations(response_format="srt")
with open("video_audio.mp3", "rb") as fh:
    srt_text = stt.transcribe(fh.read())
with open("subtitles.srt", "w") as f:
    f.write(srt_text)
```

### Per-call parameter override including file_name

```python
stt = STTPollinations(language="en")
with open("recording.wav", "rb") as fh:
    result = stt.transcribe(fh.read(), file_name="recording.wav")
```

### Inspect raw HTTP response

```python
with open("speech.mp3", "rb") as fh:
    response = stt.transcribe_response(fh.read())
print(response.headers.get("content-type"))
print(response.text)
```

### Derive variants with `with_params`

```python
base     = STTPollinations(model="whisper-large-v3")
en_stt   = base.with_params(language="en", temperature=0.0)
es_stt   = base.with_params(language="es")
verbose  = base.with_params(response_format="verbose_json")
```

### LangChain chain integration

```python
from langchain_core.runnables import RunnableLambda

load_audio = RunnableLambda(lambda path: open(path, "rb").read())
pipeline   = load_audio | STTPollinations(language="en")
result     = pipeline.invoke("speech.mp3")
print(result.text)
```

### Standalone TranscriptionParams

```python
from langchain_pollinations.stt import TranscriptionParams

p = TranscriptionParams(
    model="whisper-large-v3",
    language="fr",
    response_format="verbose_json",
    temperature=0.2,
)
print(p.to_form_data())
# {'model': 'whisper-large-v3', 'language': 'fr', 'response_format': 'verbose_json', 'temperature': '0.2'}
```

### Force audio model catalog refresh

```python
from langchain_pollinations._audio_catalog import _load_audio_model_ids

fresh_ids = _load_audio_model_ids(force=True)
print(fresh_ids)
```
