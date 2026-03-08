## Classes

### ModelInformation

`ModelInformation` is a `@dataclass(slots=True)` that provides discovery of available models from the Pollinations API.

#### Instantiation

```python
from langchain_pollinations import ModelInformation

models = ModelInformation(api_key="your-key")
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | API key. Falls back to `POLLINATIONS_API_KEY` environment variable. |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | API base URL. |
| `timeout_s` | `float` | `120.0` | HTTP timeout in seconds. |

#### Methods

##### `get_available_models() → dict[str, list[str]]`

Synchronously fetch and return model IDs for all modalities. Calls `list_text_models()`, `list_image_models()`, and `list_audio_models()` internally. Any endpoint that fails returns an empty list for that key without raising an exception.

**Returns:** `dict[str, list[str]]` with three keys: `"text"`, `"image"`, and `"audio"`.

```python
catalog = models.get_available_models()
print(catalog["text"])   # list of text model IDs
print(catalog["image"])  # list of image model IDs
print(catalog["audio"])  # list of audio model IDs
```

##### `aget_available_models() → dict[str, list[str]]`

Async version of `get_available_models`.

##### `list_compatible_models() → dict[str, Any]`

Synchronously list all OpenAI-compatible models from `GET /v1/models`.

##### `alist_compatible_models() → dict[str, Any]`

Async version of `list_compatible_models`.

##### `list_text_models() → list[dict[str, Any]] | dict[str, Any]`

Synchronously fetch full model records from `GET /text/models`.

##### `alist_text_models() → list[dict[str, Any]] | dict[str, Any]`

Async version of `list_text_models`.

##### `list_image_models() → list[dict[str, Any]] | dict[str, Any]`

Synchronously fetch full model records from `GET /image/models`.

##### `alist_image_models() → list[dict[str, Any]] | dict[str, Any]`

Async version of `list_image_models`.

##### `list_audio_models() → list[dict[str, Any]] | dict[str, Any]`

Synchronously fetch full model records from `GET /audio/models`. Covers both TTS (text-to-speech) and STT (speech-to-text) models.

##### `alist_audio_models() → list[dict[str, Any]] | dict[str, Any]`

Async version of `list_audio_models`.

#### Method Summary

| Method | Async | Endpoint | Returns |
|---|---|---|---|
| `get_available_models()` | `aget_available_models()` | `/text/models` + `/image/models` + `/audio/models` | `dict[str, list[str]]` with `"text"`, `"image"`, and `"audio"` keys |
| `list_compatible_models()` | `alist_compatible_models()` | `GET /v1/models` | `dict[str, Any]` |
| `list_text_models()` | `alist_text_models()` | `GET /text/models` | `list[dict[str, Any]] \| dict[str, Any]` |
| `list_image_models()` | `alist_image_models()` | `GET /image/models` | `list[dict[str, Any]] \| dict[str, Any]` |
| `list_audio_models()` | `alist_audio_models()` | `GET /audio/models` | `list[dict[str, Any]] \| dict[str, Any]` |

---

## Usage Examples

### List all available model IDs by modality

```python
from langchain_pollinations import ModelInformation

models = ModelInformation(api_key="your-key")
catalog = models.get_available_models()

print("Text models:",  catalog["text"])
print("Image models:", catalog["image"])
print("Audio models:", catalog["audio"])
```

### List full model records per endpoint

```python
text_records  = models.list_text_models()
image_records = models.list_image_models()
audio_records = models.list_audio_models()
compat        = models.list_compatible_models()
```

### Async usage

```python
catalog       = await models.aget_available_models()
audio_records = await models.alist_audio_models()
```
