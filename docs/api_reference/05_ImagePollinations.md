## Classes

### ImagePollinations

`ImagePollinations` is a Pydantic `BaseModel` that also implements the LangChain `Runnable[str, bytes]` interface. It wraps the Pollinations `GET /image/{prompt}` endpoint to support synchronous and asynchronous image and video generation.

#### Instantiation

```python
from langchain_pollinations import ImagePollinations

img = ImagePollinations(
    api_key="your-key",       # Optional if POLLINATIONS_API_KEY is set
    model="flux",
    width=1024,
    height=1024,
    seed=42,
    enhance=True,
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str \| None` | `None` | Optional Runnable identifier for LangChain logs and traces. |
| `api_key` | `str \| None` | `None` | API key. Falls back to `POLLINATIONS_API_KEY` environment variable. Excluded from `model_dump`. |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | API base URL. |
| `timeout_s` | `float` | `120.0` | HTTP timeout in seconds. |
| `model` | `str \| None` | `None` | Image/video model ID. `None` lets the API apply its own default. A `UserWarning` is emitted for unknown IDs. |
| `width` | `int \| None` | `None` | Output width in pixels. |
| `height` | `int \| None` | `None` | Output height in pixels. |
| `seed` | `int \| None` | `None` | Random seed (`-1` to `2147483647`). |
| `enhance` | `bool \| None` | `None` | Enable prompt enhancement. |
| `negative_prompt` | `str \| None` | `None` | Negative prompt guidance. |
| `safe` | `bool \| None` | `None` | Enable content safety filter. |
| `quality` | `Quality \| None` | `None` | `"low"` \| `"medium"` \| `"high"` \| `"hd"` (`gptimage` models only). |
| `image` | `str \| None` | `None` | Reference image URL(s); multiple URLs separated by comma or pipe (`\|`). |
| `transparent` | `bool \| None` | `None` | Transparent background (`gptimage` models only). |
| `duration` | `int \| None` | `None` | Video duration in seconds (`1`–`10`, video models only). |
| `aspect_ratio` | `str \| None` | `None` | Aspect ratio string (e.g. `"16:9"`). Serialised as `aspectRatio` in the query string. |
| `audio` | `bool \| None` | `None` | Include audio track in video (`veo` only). |

All `None` instance fields are omitted from the query string; `ImagePromptParams` fills them with its own defaults.

#### Available Models

`ImageModelId = str`. The catalog is fetched automatically from `GET /image/models` on the first instantiation. Known values in the fallback catalog:

| Type | Models |
|---|---|
| Image | `flux`, `zimage`, `turbo`, `klein`, `klein-large`, `nanobanana`, `nanobanana-pro`, `seedream`, `seedream-pro`, `kontext` |
| Image (quality tiers) | `gptimage`, `gptimage-large`, `imagen-4` |
| Video | `veo`, `grok-video`, `seedance`, `seedance-pro`, `wan`, `ltx-2` |

An unknown model ID triggers a `UserWarning` but does not block the request. To force a catalog refresh at runtime:

```python
from langchain_pollinations.image import _load_image_model_ids

_load_image_model_ids(force=True)
```

#### Methods

##### `generate(prompt, *, params=None, **kwargs) → bytes`

Synchronously generate image or video bytes.

**Parameters:**
- `prompt` (`str`): Text description of the image/video to generate.
- `params` (`dict[str, Any] | None`): Optional per-call query parameter overrides.
- `**kwargs`: Additional per-call overrides as keyword arguments.

**Returns:** Raw binary content (`bytes`) of the generated image or video.

```python
data = img.generate("a sunset over the mountains")
with open("output.jpg", "wb") as f:
    f.write(data)
```

##### `agenerate(prompt, *, params=None, **kwargs) → bytes`

Async version of `generate`.

```python
data = await img.agenerate("a sunset over the mountains")
```

##### `generate_response(prompt, *, params=None, **kwargs) → httpx.Response`

Synchronously execute the request and return the raw `httpx.Response`. Useful for inspecting `Content-Type` before writing bytes.

**Returns:** `httpx.Response`

##### `agenerate_response(prompt, *, params=None, **kwargs) → httpx.Response`

Async version of `generate_response`.

##### `invoke(input, config=None, **kwargs) → bytes`

LangChain-compatible synchronous invocation.

**Parameters:**
- `input` (`str`): Text prompt.
- `config` (`Any | None`): Accepted for interface compatibility; unused.
- `**kwargs`: Additional generation parameters forwarded to `generate`.

**Returns:** `bytes`

##### `ainvoke(input, config=None, **kwargs) → bytes`

LangChain-compatible asynchronous invocation.

**Returns:** `bytes`

##### `with_params(**overrides) → ImagePollinations`

Return a new `ImagePollinations` instance with merged configuration. The original instance is not mutated. The `api_key` is explicitly preserved in the copy.

**Parameters:**
- `**overrides`: Parameter values to override in the new instance.

**Returns:** `ImagePollinations`

```python
base = ImagePollinations(model="flux", width=1024, height=1024)
pixel_art = base.with_params(model="klein", enhance=True)
portrait   = base.with_params(width=768, height=1024, safe=True)
```

---

## ImagePromptParams

Pydantic model (`extra="forbid"`) that validates and serialises query parameters before each request. Instantiated automatically by `_build_query`. Can be used standalone:

```python
from langchain_pollinations.image import ImagePromptParams

params = ImagePromptParams(model="flux", width=512, seed=7, enhance=True)
query_dict = params.to_query()   # returns dict suitable for URL query string
```

#### Fields

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `model` | `str` | `"zimage"` | — | Model ID. |
| `width` | `int` | `1024` | `≥ 0` | Output width in pixels. |
| `height` | `int` | `1024` | `≥ 0` | Output height in pixels. |
| `seed` | `int` | `0` | `[-1, 2147483647]` | Random seed. |
| `enhance` | `bool` | `False` | — | Enable prompt enhancement. |
| `negative_prompt` | `str` | `"worst quality, blurry"` | — | Negative prompt guidance. |
| `safe` | `bool` | `False` | — | Enable content safety filter. |
| `quality` | `Quality` | `"medium"` | — | Output quality level (`gptimage` only). |
| `image` | `str \| None` | `None` | — | Reference image URL(s), comma- or pipe-separated. |
| `transparent` | `bool` | `False` | — | Transparent background (`gptimage` only). |
| `duration` | `int \| None` | `None` | `[1, 10]` | Video duration in seconds (video models only). |
| `aspect_ratio` | `str \| None` | `None` | alias: `aspectRatio` | Aspect ratio string (e.g. `"16:9"`). |
| `audio` | `bool` | `False` | — | Include audio track in video (`veo` only). |

#### `to_query() → dict[str, Any]`

Serialise the model to a dictionary suitable for use as a URL query string. Uses field aliases (`aspectRatio`) and excludes `None` fields.

---

## Type Aliases

Defined in `langchain_pollinations.image`:

```python
ImageModelId = str
Quality = Literal["low", "medium", "high", "hd"]
```

---

## Module-level Helpers

Defined in `langchain_pollinations.image`.

### `_load_image_model_ids`

```python
def _load_image_model_ids(
    api_key: str | None = None,
    *,
    force: bool = False,
) -> list[str]
```

Fetch the list of available image model IDs from `GET /image/models` and update the module-level cache. The remote call is made **at most once per process lifetime**. Subsequent calls return the cached list immediately unless `force=True` is passed. On failure the fallback catalog is retained without raising an exception.

**Parameters:**
- `api_key`: API key forwarded to `ModelInformation`. Resolved from `POLLINATIONS_API_KEY` when `None`.
- `force`: Bypass the one-shot guard and re-fetch regardless of prior calls.

**Returns:** A copy of the current (possibly freshly updated) image model ID list.

---

## API Endpoint

`GET /image/{prompt}` — the prompt is URL-encoded before being appended to the path.

---

## Usage Examples

### Basic image generation

```python
import dotenv
from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

img = ImagePollinations(model="flux", width=1024, height=1024, seed=42)
data = img.generate("a photorealistic sunset over the Andes mountains")
with open("sunset.jpg", "wb") as f:
    f.write(data)
```

### Async generation

```python
img = ImagePollinations(model="flux")
data = await img.agenerate("a futuristic cityscape at night")
```

### Inspect Content-Type before saving

```python
img = ImagePollinations(model="veo", duration=5)
response = img.generate_response("a dolphin jumping out of the ocean")
content_type = response.headers.get("content-type", "")
ext = "mp4" if "video" in content_type else "jpg"
with open(f"output.{ext}", "wb") as f:
    f.write(response.content)
```

### Video generation

```python
img = ImagePollinations(model="seedance", duration=5, aspect_ratio="16:9")
data = img.generate("a timelapse of clouds over a mountain range")
with open("output.mp4", "wb") as f:
    f.write(data)
```

### Quality-tier image (gptimage)

```python
img = ImagePollinations(model="gptimage", quality="hd", transparent=False)
data = img.generate("a professional product photo of a coffee cup")
```

### Per-call parameter overrides via `params`

```python
img = ImagePollinations(model="flux", width=1024, height=1024)
data = img.generate(
    "a macro photo of a butterfly",
    params={"seed": 99, "enhance": True, "negative_prompt": "blurry, low quality"},
)
```

### Derive variants with `with_params`

```python
base      = ImagePollinations(model="flux", width=1024, height=1024)
square    = base.with_params(width=512, height=512)
landscape = base.with_params(width=1920, height=1080, enhance=True)
safe_mode = base.with_params(safe=True, negative_prompt="nsfw")
```

### LangChain chain integration

```python
from langchain_core.runnables import RunnableLambda

extract_prompt = RunnableLambda(lambda x: x["prompt"])
pipeline = extract_prompt | ImagePollinations(model="flux")
data = pipeline.invoke({"prompt": "a serene Japanese garden"})
```

### Standalone `ImagePromptParams`

```python
from langchain_pollinations.image import ImagePromptParams

params = ImagePromptParams(
    model="gptimage",
    width=1024,
    height=1024,
    quality="hd",
    seed=0,
)
print(params.to_query())
# {'model': 'gptimage', 'width': 1024, 'height': 1024, 'quality': 'hd', 'seed': 0, ...}
```

### Force catalog refresh

```python
from langchain_pollinations.image import _load_image_model_ids

fresh_ids = _load_image_model_ids(force=True)
print(fresh_ids)
```
