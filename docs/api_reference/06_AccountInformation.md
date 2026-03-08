## Classes

### AccountInformation

`AccountInformation` is a `@dataclass(slots=True)` that provides access to account profile, balance, API key metadata, and usage statistics.

#### Instantiation

```python
from langchain_pollinations import AccountInformation

account = AccountInformation(api_key="your-key")
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | API key. Falls back to `POLLINATIONS_API_KEY` environment variable. |
| `base_url` | `str` | `"https://gen.pollinations.ai"` | API base URL. |
| `timeout_s` | `float` | `120.0` | HTTP timeout in seconds. |

#### Methods

##### `get_profile() → AccountProfile`

Synchronously fetch the account profile from `GET /account/profile`.

**Returns:** `AccountProfile` instance with all profile fields.

##### `aget_profile() → AccountProfile`

Async version of `get_profile`.

##### `get_balance() → dict[str, Any]`

Synchronously fetch account balance from `GET /account/balance`.

##### `aget_balance() → dict[str, Any]`

Async version of `get_balance`.

##### `get_key() → dict[str, Any]`

Synchronously fetch the current API key metadata from `GET /account/key`.

##### `aget_key() → dict[str, Any]`

Async version of `get_key`.

##### `get_usage(params: AccountUsageParams | None = None) → AccountUsageResponse | str`

Synchronously fetch detailed usage records from `GET /account/usage`.

When `params.format` is `"json"` (the default), the response is validated and returned as an `AccountUsageResponse` instance. When `params.format` is `"csv"`, the raw CSV text is returned as `str`.

```python
from langchain_pollinations.account import AccountUsageParams

usage = account.get_usage(AccountUsageParams(limit=50, format="json"))
for record in usage.usage:
    print(record.timestamp, record.model, record.cost_usd)
```

##### `aget_usage(params: AccountUsageParams | None = None) → AccountUsageResponse | str`

Async version of `get_usage`.

##### `get_usage_daily(params: AccountUsageDailyParams | None = None) → dict[str, Any] | str`

Synchronously fetch daily aggregated usage statistics from `GET /account/usage/daily`.

**Returns:** `dict[str, Any]` for JSON format or `str` for CSV format.

##### `aget_usage_daily(params: AccountUsageDailyParams | None = None) → dict[str, Any] | str`

Async version of `get_usage_daily`.

#### API Endpoint Mapping

| Method | Async | Endpoint |
|---|---|---|
| `get_profile` | `aget_profile` | `GET /account/profile` |
| `get_balance` | `aget_balance` | `GET /account/balance` |
| `get_key` | `aget_key` | `GET /account/key` |
| `get_usage` | `aget_usage` | `GET /account/usage` |
| `get_usage_daily` | `aget_usage_daily` | `GET /account/usage/daily` |

---

## Account Data Models

Defined in `langchain_pollinations.account`. All Pydantic models unless stated otherwise.

### Type Aliases

```python
AccountFormat   = Literal["json", "csv"]
AccountTier     = Literal["anonymous", "microbe", "spore", "seed", "flower", "nectar", "router"]
ApiKeyType      = str    # known values: "secret" | "publishable"
MeterSource     = str    # known values: "tier" | "pack" | "crypto"
Limit1To50000   = Annotated[int, Field(ge=1, le=50000)]
```

`ApiKeyType` and `MeterSource` are typed as plain `str` to tolerate future values without raising a validation error.

### AccountProfile

Pydantic model for `GET /account/profile`. `model_config = ConfigDict(extra="allow", populate_by_name=True)`. camelCase API fields are mapped to snake_case via `Field(alias=...)`.

| Field | Type | API alias | Description |
|---|---|---|---|
| `name` | `str \| None` | — | Display name. |
| `email` | `str \| None` | — | Account email. |
| `github_username` | `str \| None` | `githubUsername` | Linked GitHub username. |
| `image` | `str \| None` | — | Profile picture URL. |
| `tier` | `AccountTier` | — | Subscription tier. |
| `created_at` | `str` | `createdAt` | ISO 8601 creation timestamp. |
| `next_reset_at` | `str \| None` | `nextResetAt` | ISO 8601 next daily reset timestamp. `None` for accounts without a reset cycle. |

### AccountUsageRecord

Pydantic model for one entry in the usage array from `GET /account/usage`. `model_config = ConfigDict(extra="allow")`. Token counts use `float` to match the JSON `number` type.

| Field | Type | Default | Description |
|---|---|---|---|
| `timestamp` | `str` | — | `YYYY-MM-DD HH:mm:ss` request timestamp. |
| `type` | `str` | — | Request type. Known values: `"generate.text"`, `"generate.image"`. |
| `model` | `str \| None` | `None` | Model used. |
| `api_key` | `str \| None` | `None` | Masked key identifier. |
| `api_key_type` | `ApiKeyType \| None` | `None` | Key category. Known values: `"secret"`, `"publishable"`. |
| `meter_source` | `MeterSource \| None` | `None` | Billing source. Known values: `"tier"`, `"pack"`, `"crypto"`. |
| `input_text_tokens` | `float` | `0.0` | Prompt text token count. |
| `input_cached_tokens` | `float` | `0.0` | Cache-hit token count. |
| `input_audio_tokens` | `float` | `0.0` | Prompt audio token count. |
| `input_image_tokens` | `float` | `0.0` | Prompt image token count. |
| `output_text_tokens` | `float` | `0.0` | Completion text token count. |
| `output_reasoning_tokens` | `float` | `0.0` | Internal reasoning token count. |
| `output_audio_tokens` | `float` | `0.0` | Completion audio token count. |
| `output_image_tokens` | `float` | `0.0` | Generated images (1 per image). |
| `cost_usd` | `float` | `0.0` | Request cost in USD. |
| `response_time_ms` | `float \| None` | `None` | End-to-end latency in ms. |

### AccountUsageResponse

Pydantic model for the full body of `GET /account/usage`. `model_config = ConfigDict(extra="allow")`.

```python
class AccountUsageResponse(BaseModel):
    usage: list[AccountUsageRecord]
    count: int
```

### AccountUsageParams

Pydantic model for `GET /account/usage` query parameters. `model_config = ConfigDict(extra="forbid")`.

```python
class AccountUsageParams(BaseModel):
    format: AccountFormat = "json"
    limit: Limit1To50000 = 100      # ge=1, le=50000
    before: str | None = None       # ISO timestamp cursor for pagination
```

### AccountUsageDailyParams

Pydantic model for `GET /account/usage/daily` query parameters. `model_config = ConfigDict(extra="forbid")`.

```python
class AccountUsageDailyParams(BaseModel):
    format: AccountFormat = "json"
```

---

## Usage Examples

### Profile

```python
from langchain_pollinations import AccountInformation

account = AccountInformation(api_key="your-key")
profile = account.get_profile()
print(profile.tier, profile.email, profile.next_reset_at)
```

### Balance

```python
balance = account.get_balance()
print(balance)
```

### API key metadata

```python
key_info = account.get_key()
print(key_info)
```

### Usage records (JSON)

```python
from langchain_pollinations.account import AccountUsageParams

usage = account.get_usage(AccountUsageParams(limit=50, format="json"))
for record in usage.usage:
    print(record.timestamp, record.model, record.cost_usd)
print("total records:", usage.count)
```

### Usage records (CSV)

```python
csv_text = account.get_usage(AccountUsageParams(format="csv", limit=200))
print(csv_text)
```

### Usage records with pagination cursor

```python
page1 = account.get_usage(AccountUsageParams(limit=100))
if page1.usage:
    cursor = page1.usage[-1].timestamp
    page2 = account.get_usage(AccountUsageParams(limit=100, before=cursor))
```

### Daily usage

```python
from langchain_pollinations.account import AccountUsageDailyParams

daily = account.get_usage_daily(AccountUsageDailyParams(format="json"))
print(daily)
```

### Async usage

```python
profile = await account.aget_profile()
usage   = await account.aget_usage(AccountUsageParams(limit=25))
```
