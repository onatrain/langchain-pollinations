## Classes

### ImagePollinations

Wrapper for `GET /image/{prompt}`. Supports image and video generation synchronously and asynchronously.

#### Instantiation

```python
from langchain_pollinations import ImagePollinations

img = ImagePollinations(
    api_key="your-key",
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
| `api_key` | `str \| None` | `None` | API key. |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | API base URL. |
| `timeout_s` | `float` | `120.0` | HTTP timeout in seconds. |
| `model` | `str \| None` | `None` | Image/video model ID. |
| `width` | `int \| None` | `None` | Output width in pixels. |
| `height` | `int \| None` | `None` | Output height in pixels. |
| `seed` | `int \| None` | `None` | Random seed (`-1` to `2147483647`). |
| `enhance` | `bool \| None` | `None` | Enable prompt enhancement. |
| `negative_prompt` | `str \| None` | `None` | Negative prompt guidance. |
| `safe` | `bool \| None` | `None` | Enable content safety filter. |
| `quality` | `Quality \| None` | `None` | `"low"` \| `"medium"` \| `"high"` \| `"hd"` (gptimage only). |
| `image` | `str \| None` | `None` | Reference image URLs (comma- or pipe-separated). |
| `transparent` | `bool \| None` | `None` | Transparent background (gptimage only). |
| `duration` | `int \| None` | `None` | Video duration in seconds (`1`–`10`, video models only). |
| `aspect_ratio` | `str \| None` | `None` | Aspect ratio string (e.g. `"16:9"`). Serialised as `aspectRatio`. |
| `audio` | `bool \| None` | `None` | Include audio track in video (veo only). |

#### Available Models

`ImageModelId = str`. Known values at release time:

| Type | Models |
|---|---|
| Image | `flux`, `zimage`, `turbo`, `klein`, `klein-large`, `nanobanana`, `nanobanana-pro`, `seedream`, `seedream-pro`, `kontext` |
| Image (quality) | `gptimage`, `gptimage-large`, `imagen-4` |
| Video | `veo`, `grok-video`, `seedance`, `seedance-pro`, `wan`, `ltx-2` |

#### Methods

##### `generate(prompt, *, params=None, **kwargs) → bytes`

Synchronously generate and return raw image/video bytes.

##### `agenerate(prompt, *, params=None, **kwargs) → bytes`

Async version of `generate`.

##### `generate_response(prompt, *, params=None, **kwargs) → httpx.Response`

Return the raw HTTP response. Useful for inspecting `Content-Type` before writing.

##### `agenerate_response(prompt, *, params=None, **kwargs) → httpx.Response`

Async version of `generate_response`.

##### `invoke(input, config=None, **kwargs) → bytes`

LangChain-compatible synchronous invocation.

##### `ainvoke(input, config=None, **kwargs) → bytes`

LangChain-compatible asynchronous invocation.

##### `with_params(**overrides) → ImagePollinations`

Return a new `ImagePollinations` instance with merged configuration. The original instance is not mutated.

```python
base = ImagePollinations(model="flux", width=1024, height=1024)
pixel_art = base.with_params(model="klein", enhance=True)
portrait  = base.with_params(width=768, height=1024, safe=True)
```

#### ImagePromptParams

Internal Pydantic model validating and serialising query parameters before each request. Instantiated automatically by `_build_query`. Can be used standalone:

```python
from langchain_pollinations.image import ImagePromptParams

params = ImagePromptParams(model="flux", width=512, seed=7, enhance=True)
query_dict = params.to_query()   # returns dict suitable for URL query string
```

Default values used when an `ImagePollinations` instance does not override them:

| Field | Default |
|---|---|
| `model` | `"zimage"` |
| `width` | `1024` |
| `height` | `1024` |
| `seed` | `0` |
| `enhance` | `False` |
| `negative_prompt` | `"worst quality, blurry"` |
| `safe` | `False` |
| `quality` | `"medium"` |
| `transparent` | `False` |
| `audio` | `False` |

#### API Endpoint

`GET /image/{prompt}` — prompt is URL-encoded.
