## Classes

### ModelInformation

Discovery of available models.

#### Instantiation

```python
from langchain_pollinations import ModelInformation

models = ModelInformation(api_key="your-key")
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | API key. |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | API base URL. |
| `timeout_s` | `float` | `120.0` | HTTP timeout in seconds. |

#### Methods

| Method | Async | Endpoint | Returns |
|---|---|---|---|
| `get_available_models()` | `aget_available_models()` | `/text/models` + `/image/models` | `dict[str, list[str]]` with `"text"` and `"image"` keys |
| `list_compatible_models()` | `alist_compatible_models()` | `GET /v1/models` | `dict[str, Any]` |
| `list_text_models()` | `alist_text_models()` | `GET /text/models` | `list[dict[str, Any]] \| dict[str, Any]` |
| `list_image_models()` | `alist_image_models()` | `GET /image/models` | `list[dict[str, Any]] \| dict[str, Any]` |
