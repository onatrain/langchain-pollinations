## Classes

### PollinationsAPIError

Raised by all classes when an API request fails. Defined in `langchain_pollinations._errors` and exported from `langchain_pollinations`.

```python
@dataclass(slots=True)
class PollinationsAPIError(RuntimeError):
    status_code: int
    message: str
    body: str | None = None
    error_code: str | None = None     # "BAD_REQUEST" | "UNAUTHORIZED" | "FORBIDDEN"
                                       # | "PAYMENT_REQUIRED" | "INTERNAL_ERROR"
    request_id: str | None = None
    timestamp: str | None = None
    details: dict[str, Any] | None = None
    cause: Any | None = None
```

#### Properties

| Property | Returns | Condition |
|---|---|---|
| `is_client_error` | `bool` | `400 ≤ status_code < 500` |
| `is_server_error` | `bool` | `500 ≤ status_code < 600` |
| `is_auth_error` | `bool` | `status_code in (401, 403)` |
| `is_validation_error` | `bool` | `status_code == 400` and `error_code == "BAD_REQUEST"` |
| `is_payment_required` | `bool` | `status_code == 402` or `error_code == "PAYMENT_REQUIRED"` |

#### Methods

##### `to_dict() → dict[str, Any]`

Serialize all fields for structured logging.

#### Error codes

| HTTP | `error_code` | Meaning |
|---|---|---|
| `400` | `BAD_REQUEST` | Invalid parameters or schema validation failure. |
| `401` | `UNAUTHORIZED` | Missing or invalid API key. |
| `402` | `PAYMENT_REQUIRED` | Insufficient Pollen balance. |
| `403` | `FORBIDDEN` | Valid key, insufficient permission scope. |
| `500` | `INTERNAL_ERROR` | Unhandled server-side failure. |

#### Error handling example

```python
from langchain_pollinations import ChatPollinations, PollinationsAPIError
from langchain_core.messages import HumanMessage

try:
    llm = ChatPollinations(model="openai")
    res = llm.invoke([HumanMessage(content="Hello")])
except PollinationsAPIError as e:
    if e.is_auth_error:
        print("Check your POLLINATIONS_API_KEY.")
    elif e.is_payment_required:
        print("Pollen balance exhausted.")
    elif e.is_validation_error:
        print(f"Bad request: {e.details}")
    elif e.is_server_error:
        print(f"Server error {e.status_code} — consider retrying.")
    else:
        print(e.to_dict())
```
