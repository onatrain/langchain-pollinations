from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from langchain_pollinations._auth import AuthConfig
from langchain_pollinations._client import HttpConfig, PollinationsHttpClient

DEFAULT_BASE_URL = "https://gen.pollinations.ai"

AccountFormat = Literal["json", "csv"]
Limit1To50000 = Annotated[int, Field(ge=1, le=50000)]


class AccountUsageParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: AccountFormat = "json"
    limit: Limit1To50000 = 100
    before: Optional[str] = None


class AccountUsageDailyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: AccountFormat = "json"


@dataclass(slots=True)
class AccountInformation:
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 120.0

    _http: PollinationsHttpClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        auth = AuthConfig.from_env_or_value(self.api_key)
        self._http = PollinationsHttpClient(
            config=HttpConfig(base_url=self.base_url, timeout_s=self.timeout_s),
            api_key=auth.api_key,
        )

    def get_profile(self) -> dict[str, Any]:
        return self._http.get("/account/profile").json()

    def get_balance(self) -> dict[str, Any]:
        return self._http.get("/account/balance").json()

    def get_key(self) -> dict[str, Any]:
        return self._http.get("/account/key").json()

    def get_usage(self, params: AccountUsageParams | None = None) -> dict[str, Any]:
        q = (params or AccountUsageParams()).model_dump(exclude_none=True)
        return self._http.get("/account/usage", params=q).json()

    def get_usage_daily(self, params: AccountUsageDailyParams | None = None) -> dict[str, Any]:
        q = (params or AccountUsageDailyParams()).model_dump(exclude_none=True)
        return self._http.get("/account/usage/daily", params=q).json()
