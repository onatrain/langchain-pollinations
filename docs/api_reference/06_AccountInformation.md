## Classes

### AccountInformation

Access to account profile, balance, API key metadata, and usage statistics.

#### Instantiation

```python
from langchain_pollinations import AccountInformation

account = AccountInformation(api_key="your-key")
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | API key. |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | API base URL. |
| `timeout_s` | `float` | `120.0` | HTTP timeout in seconds. |

#### Methods

##### `get_profile() → AccountProfile`:

Serves `GET /account/profile`. Returns a typed `AccountProfile` instance.

##### `aget_profile() → AccountProfile`:

Async version.

##### `get_balance() → dict[str, Any]`:

Serves `GET /account/balance`.

##### `aget_balance() → dict[str, Any]`:

Async version.

##### `get_key() → dict[str, Any]`:

Serves `GET /account/key`.

##### `aget_key() → dict[str, Any]`:

Async version.

##### `get_usage(params=None) → AccountUsageResponse | str`:

Serves `GET /account/usage`. Returns `AccountUsageResponse` for JSON format or `str` for CSV.

```python
from langchain_pollinations.account import AccountUsageParams

usage = account.get_usage(AccountUsageParams(limit=50, format="json"))
for record in usage.usage:
    print(record.timestamp, record.model, record.cost_usd)
```

##### `aget_usage(params=None) → AccountUsageResponse | str`:

Async version.

##### `get_usage_daily(params=None) → dict[str, Any] | str`:

Serves `GET /account/usage/daily`.

##### `aget_usage_daily(params=None) → dict[str, Any] | str`:

Async version.

#### API Endpoint Mapping

| Method | Endpoint |
|---|---|
| `get_profile` | `GET /account/profile` |
| `get_balance` | `GET /account/balance` |
| `get_key` | `GET /account/key` |
| `get_usage` | `GET /account/usage` |
| `get_usage_daily` | `GET /account/usage/daily` |

---

## Account Data Models

Defined in `langchain_pollinations.account` and also exported from `langchain_pollinations`.

### AccountTier

```python
AccountTier = Literal["anonymous", "microbe", "spore", "seed", "flower", "nectar", "router"]
```

### AccountProfile

Pydantic model for `GET /account/profile`. camelCase API fields are mapped to snake_case:

| Field | Type | API alias | Description |
|---|---|---|---|
| `name` | `str \| None` | — | Display name. |
| `email` | `str \| None` | — | Account email. |
| `github_username` | `str \| None` | `githubUsername` | Linked GitHub username. |
| `image` | `str \| None` | — | Profile picture URL. |
| `tier` | `AccountTier` | — | Subscription tier. |
| `created_at` | `str` | `createdAt` | ISO 8601 creation timestamp. |
| `next_reset_at` | `str \| None` | `nextResetAt` | ISO 8601 next daily reset timestamp. |

### AccountUsageRecord

Pydantic model for one entry in the usage list:

| Field | Type | Description |
|---|---|---|
| `timestamp` | `str` | `YYYY-MM-DD HH:mm:ss` request timestamp. |
| `type` | `str` | Request type (`"generate.text"`, `"generate.image"`). |
| `model` | `str \| None` | Model used. |
| `api_key` | `str \| None` | Masked key identifier. |
| `api_key_type` | `str \| None` | Key category (`"secret"`, `"publishable"`). |
| `meter_source` | `str \| None` | Billing source (`"tier"`, `"pack"`, `"crypto"`). |
| `input_text_tokens` | `float` | Prompt text token count. |
| `input_cached_tokens` | `float` | Cache-hit token count. |
| `input_audio_tokens` | `float` | Prompt audio token count. |
| `input_image_tokens` | `float` | Prompt image token count. |
| `output_text_tokens` | `float` | Completion text token count. |
| `output_reasoning_tokens` | `float` | Internal reasoning token count. |
| `output_audio_tokens` | `float` | Completion audio token count. |
| `output_image_tokens` | `float` | Generated images (1 per image). |
| `cost_usd` | `float` | Request cost in USD. |
| `response_time_ms` | `float \| None` | End-to-end latency in ms. |

### AccountUsageResponse

```python
class AccountUsageResponse(BaseModel):
    usage: list[AccountUsageRecord]
    count: int
```

### AccountUsageParams

```python
class AccountUsageParams(BaseModel):
    format: Literal["json", "csv"] = "json"
    limit: int = 100               
    before: str | None = None      # ISO timestamp cursor for pagination
```

### AccountUsageDailyParams

```python
class AccountUsageDailyParams(BaseModel):
    format: Literal["json", "csv"] = "json"
```
