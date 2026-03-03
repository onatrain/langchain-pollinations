## Classes

### ChatPollinationsConfig

Pydantic model for all request-level parameters of the chat completions endpoint. Used as `request_defaults` in `ChatPollinations` or passed as loose kwargs.

```python
from langchain_pollinations.chat import ChatPollinationsConfig
```

#### Fields

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `model` | `str \| None` | `None` | — | Model ID. A `UserWarning` is emitted for unknown IDs. |
| `modalities` | `list[Modality] \| None` | `None` | — | Output modalities: `"text"` and/or `"audio"`. |
| `audio` | `AudioConfig \| None` | `None` | — | Audio generation config (voice + format). Required when `"audio"` is in `modalities`. |
| `temperature` | `float \| None` | `None` | `[0, 2]` | Sampling temperature. |
| `top_p` | `float \| None` | `None` | `[0, 1]` | Nucleus sampling parameter. |
| `max_tokens` | `int \| None` | `None` | `[0, 9007199254740991]` | Maximum tokens to generate. |
| `stop` | `str \| list[str] \| None` | `None` | list: 1–4 items | Stop sequences. |
| `seed` | `int \| None` | `None` | `[-1, 9007199254740991]` | Random seed for reproducibility. |
| `presence_penalty` | `float \| None` | `None` | `[-2, 2]` | Penalize tokens already present in context. |
| `frequency_penalty` | `float \| None` | `None` | `[-2, 2]` | Penalize frequently occurring tokens. |
| `repetition_penalty` | `float \| None` | `None` | `[0, 2]` | Penalize token repetition (provider-specific). |
| `logit_bias` | `dict[str, int] \| None` | `None` | values: `[-9007199254740991, 9007199254740991]` | Token ID bias map. |
| `logprobs` | `bool \| None` | `None` | — | Return log probabilities. |
| `top_logprobs` | `int \| None` | `None` | `[0, 20]` | Number of top log probabilities to return. |
| `stream` | `bool \| None` | `None` | — | Enable streaming (managed automatically by the wrapper). |
| `stream_options` | `StreamOptions \| None` | `None` | — | Streaming behavior options. |
| `response_format` | `ResponseFormat \| None` | `None` | — | Structured output format. |
| `tools` | `list[ToolDef] \| None` | `None` | — | Tool definitions for function calling. |
| `tool_choice` | `ToolChoice \| None` | `None` | — | Tool selection strategy. |
| `parallel_tool_calls` | `bool \| None` | `None` | — | Allow parallel tool calls. |
| `user` | `str \| None` | `None` | — | Caller-defined user identifier. |
| `functions` | `list[FunctionDef] \| None` | `None` | 1–128 items | Legacy function definitions. |
| `function_call` | `FunctionCall \| None` | `None` | — | Legacy function call mode. |
| `thinking` | `ThinkingConfig \| None` | `None` | — | Internal reasoning configuration. |
| `reasoning_effort` | `ReasoningEffort \| None` | `None` | — | Reasoning effort level for supported models. |
| `thinking_budget` | `int \| None` | `None` | `[0, 9007199254740991]` | Token budget for reasoning (alternative to `ThinkingConfig.budget_tokens`). |
