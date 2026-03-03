## Type Aliases and Primitive Types

Defined in `langchain_pollinations.chat`:

```python
TextModelId = str
Modality = Literal["text", "audio"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
AudioFormat = Literal["wav", "mp3", "flac", "opus", "pcm16"]
```

`VoiceId` is a `Literal` union of all supported voice identifiers:
`"alloy"`, `"echo"`, `"fable"`, `"onyx"`, `"shimmer"`, `"coral"`, `"verse"`, `"ballad"`, `"ash"`, `"sage"`, `"amuch"`, `"dan"`, `"rachel"`, `"domi"`, `"bella"`, `"elli"`, `"charlotte"`, `"dorothy"`, `"sarah"`, `"emily"`, `"lily"`, `"matilda"`, `"adam"`, `"antoni"`, `"arnold"`, `"josh"`, `"sam"`, `"daniel"`, `"charlie"`, `"james"`, `"fin"`, `"callum"`, `"liam"`, `"george"`, `"brian"`, `"bill"`.

Annotated constraint aliases:

| Alias | Base type | Constraint |
|---|---|---|
| `FloatPenalty` | `float` | `[-2, 2]` |
| `FloatRepetitionPenalty` | `float` | `[0, 2]` |
| `FloatTemperature` | `float` | `[0, 2]` |
| `FloatTopP` | `float` | `[0, 1]` |
| `Int0ToInt53` | `int` | `[0, 9007199254740991]` |
| `SeedInt` | `int` | `[-1, 9007199254740991]` |
| `TopLogprobsInt` | `int` | `[0, 20]` |
| `BiasValue` | `int` | `[-9007199254740991, 9007199254740991]` |

---

## Chat Request Configuration Models

All Pydantic models defined in `langchain_pollinations.chat`.

### AudioConfig

Configuration for audio output generation.

```python
class AudioConfig(BaseModel):
    voice: VoiceId
    format: AudioFormat   # "wav" | "mp3" | "flac" | "opus" | "pcm16"
```

Both fields are required.

### StreamOptions

```python
class StreamOptions(BaseModel):
    include_usage: bool | None = None
```
### ThinkingConfig

```python
class ThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled"] = "disabled"
    budget_tokens: int | None = None
```

### ResponseFormat

Union type selecting one of three response format schemas:

```python
ResponseFormat = Union[ResponseFormatText, ResponseFormatJsonSchema, ResponseFormatJsonObject]
```

#### ResponseFormatText

```python
class ResponseFormatText(BaseModel):
    type: Literal["text"]
```

#### ResponseFormatJsonObject

```python
class ResponseFormatJsonObject(BaseModel):
    type: Literal["json_object"]
```

#### ResponseFormatJsonSchema

```python
class ResponseFormatJsonSchema(BaseModel):
    type: Literal["json_schema"]
    json_schema: ResponseFormatJsonSchemaObject
```

#### ResponseFormatJsonSchemaObject

```python
class ResponseFormatJsonSchemaObject(BaseModel):
    description: str | None = None
    name: str | None = None
    json_schema: dict[str, Any]   # serialized as "schema" (alias)
    strict: bool | None = False
```

The field alias for `json_schema` is `"schema"`.

### ToolDef

Union selecting between a user-defined function tool and a platform builtin:

```python
ToolDef = Union[ToolFunctionTool, ToolBuiltinTool]
```

#### ToolFunctionTool

```python
class ToolFunctionTool(BaseModel):
    type: Literal["function"]
    function: ToolFunction
```

#### ToolFunction

```python
class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = {}
    strict: bool | None = False
```

#### ToolBuiltinTool

```python
class ToolBuiltinTool(BaseModel):
    type: BuiltinToolType
```

```python
BuiltinToolType = Literal[
    "code_execution",
    "google_search",
    "google_maps",
    "url_context",
    "computer_use",
    "file_search",
]
```

### ToolChoice

```python
ToolChoice = Union[Literal["none", "auto", "required"], ToolChoiceFunction]
```

#### ToolChoiceFunction

Forces the model to call a specific function by name:

```python
class ToolChoiceFunction(BaseModel):
    type: Literal["function"]
    function: ToolChoiceFunctionInner   # {"name": str}
```

### FunctionDef (legacy)

```python
class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = {}
```

### FunctionCall (legacy)

```python
FunctionCall = Union[Literal["none", "auto"], FunctionCallName]

class FunctionCallName(BaseModel):
    name: str
```

---

## Chat Message Content Block Types

Defined in `langchain_pollinations._openai_compat`. These `TypedDict` types represent the individual parts of a multipart message content list.

```python
ContentBlock = (
    ContentBlockText
    | ContentBlockImageUrl
    | ContentBlockInputAudio
    | ContentBlockVideoUrl
    | ContentBlockFile
    | ContentBlockThinking
    | ContentBlockRedactedThinking
    | dict[str, Any]
)
```

### ContentBlockText

```python
class ContentBlockText(TypedDict):
    type: Literal["text"]
    text: str
```

### ContentBlockImageUrl

```python
class ContentBlockImageUrl(TypedDict):
    type: Literal["image_url"]
    image_url: dict[str, Any]
    # image_url fields: {url: str, detail?: "auto"|"low"|"high", mime_type?: str}
```

### ContentBlockInputAudio

```python
class ContentBlockInputAudio(TypedDict):
    type: Literal["input_audio"]
    input_audio: dict[str, Any]  # {data: str, format: str}
```

### ContentBlockVideoUrl

```python
class ContentBlockVideoUrl(TypedDict):
    type: Literal["video_url"]
    video_url: dict[str, Any]
    # video_url fields: {url: str, mime_type?: str}
```

Example:

```python
{
    "type": "video_url",
    "video_url": {"url": "https://example.com/clip.mp4", "mime_type": "video/mp4"}
}
```

### ContentBlockFile

`total=False` — all keys are optional except `type`.

```python
class ContentBlockFile(TypedDict, total=False):
    type: Literal["file"]
    file: dict[str, Any]
    # file fields: {file_data?, file_id?, file_name?, file_url?, mime_type?}
    cache_control: dict[str, Any]   # e.g. {"type": "ephemeral"}
```

A flat variant is also accepted and normalized internally:

```python
# flat variant (auto-normalized to canonical form)
{"type": "file", "file_url": "https://example.com/doc.pdf", "mime_type": "application/pdf"}
```

### ContentBlockThinking

```python
class ContentBlockThinking(TypedDict):
    type: Literal["thinking"]
    thinking: str
```

Produced by reasoning-capable models (e.g. `deepseek`, `claude`).

### ContentBlockRedactedThinking

```python
class ContentBlockRedactedThinking(TypedDict):
    type: Literal["redacted_thinking"]
    data: str
```

Carries redacted reasoning traces for compliance.

---

## Chat Response Types

Defined in `langchain_pollinations._openai_compat`.

### ChatCompletionResponse

`TypedDict` mapping the raw JSON body of a `/v1/chat/completions` response.

Required fields (always present):

```python
class ChatCompletionResponse(TypedDict):   # simplified
    id: str
    object: str
    created: int        # Unix timestamp
    model: str
    choices: list[dict[str, Any]]
```

Optional fields (`total=False`):

| Field | Type | Description |
|---|---|---|
| `system_fingerprint` | `str` | Backend fingerprint. |
| `usage` | `dict[str, Any]` | Token usage statistics. |
| `user_tier` | `UserTier` | Subscription tier of the caller. |
| `citations` | `list[str]` | Source URLs from search-enabled models. |
| `prompt_filter_results` | `list[PromptFilterResultItem]` | Content moderation results for prompt. |

#### UserTier

```python
UserTier = Literal["anonymous", "spore", "seed", "flower", "nectar"]
```

Exposed in `response_metadata` of the returned `AIMessage`.

### AudioTranscript

`TypedDict` (`total=False`) for audio output data in `AIMessage.additional_kwargs["audio"]`:

```python
class AudioTranscript(TypedDict, total=False):
    transcript: str    # text transcript of the audio
    data: str          # base64-encoded audio data
    id: str            # audio object ID
    expires_at: int    # Unix expiration timestamp
```

### ContentFilterDetail

```python
class ContentFilterDetail(TypedDict):
    filtered: bool
    severity: Literal["safe", "low", "medium", "high"] | None
    detected: bool | None
```

### ContentFilterResult

`TypedDict` (`total=False`) with per-category safety results:

```python
class ContentFilterResult(TypedDict, total=False):
    hate: ContentFilterDetail
    self_harm: ContentFilterDetail
    sexual: ContentFilterDetail
    violence: ContentFilterDetail
    jailbreak: ContentFilterDetail
    protected_material_text: ContentFilterDetail
    protected_material_code: ContentFilterDetail
```

### PromptFilterResultItem

Associates a `ContentFilterResult` with a prompt index:

```python
class PromptFilterResultItem(TypedDict):
    prompt_index: int
    content_filter_results: ContentFilterResult
```

The list of `PromptFilterResultItem` objects is exposed in `AIMessage.additional_kwargs["prompt_filter_results"]`.
