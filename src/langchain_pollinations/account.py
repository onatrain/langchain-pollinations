"""
This module provides classes for managing and retrieving Pollinations account information.
It includes support for profile details, balance checks, API keys, and detailed usage statistics.
"""

from __future__ import annotations

import httpx

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from langchain_pollinations._auth import AuthConfig
from langchain_pollinations._client import HttpConfig, PollinationsHttpClient

DEFAULT_BASE_URL = "https://gen.pollinations.ai"

AccountFormat = Literal["json", "csv"]
Limit1To50000 = Annotated[int, Field(ge=1, le=50000)]


class AccountUsageParams(BaseModel):
    """
    Configuration parameters for fetching detailed account usage logs.
    Supports filtering by date, limit, and output format (JSON or CSV).
    """
    model_config = ConfigDict(extra="forbid")
    format: AccountFormat = "json"
    limit: Limit1To50000 = 100
    before: Optional[str] = None


class AccountUsageDailyParams(BaseModel):
    """
    Configuration parameters for fetching daily aggregated account usage.
    Allows specifying the desired output format for the daily usage report.
    """
    model_config = ConfigDict(extra="forbid")
    format: AccountFormat = "json"


@dataclass(slots=True)
class AccountInformation:
    """
    Main interface for interacting with Pollinations account-related endpoints.
    Provides synchronous and asynchronous methods to access profile, balance, and usage data.
    """
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 120.0

    _http: PollinationsHttpClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Initialize the HTTP client and authentication configuration after dataclass initialization.
        """
        auth = AuthConfig.from_env_or_value(self.api_key)
        self._http = PollinationsHttpClient(
            config=HttpConfig(base_url=self.base_url, timeout_s=self.timeout_s),
            api_key=auth.api_key,
        )

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict[str, Any] | str:
        """
        Parse the HTTP response based on its Content-Type header.

        Args:
            response: The raw HTTP response object from the client.

        Returns:
            A dictionary for JSON responses or a string for CSV/text responses.
        """
        content_type = response.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            return response.json()
        elif "text/csv" in content_type or "text/plain" in content_type:
            return response.text
        else:
            try:
                return response.json()
            except Exception:
                return response.text

    def get_profile(self) -> dict[str, Any]:
        """
        Retrieve the account profile information synchronously.

        Returns:
            A dictionary containing profile details such as email and user tier.
        """
        return self._http.get("/account/profile").json()

    def get_balance(self) -> dict[str, Any]:
        """
        Check the current account balance synchronously.

        Returns:
            A dictionary containing balance information and currency details.
        """
        return self._http.get("/account/balance").json()

    def get_key(self) -> dict[str, Any]:
        """
        Retrieve the current API key information for the account.

        Returns:
            A dictionary containing the API key and its metadata.
        """
        return self._http.get("/account/key").json()

    def get_usage(self, params: AccountUsageParams | None = None) -> dict[str, Any] | str:
        """
        Fetch detailed account usage records synchronously.

        Args:
            params: Optional parameters for filtering and formatting the usage data.

        Returns:
            The usage data in the requested format (JSON dictionary or CSV string).
        """
        q = (params or AccountUsageParams()).model_dump(exclude_none=True)
        response = self._http.get("/account/usage", params=q)
        return self._parse_response(response)

    def get_usage_daily(self, params: AccountUsageDailyParams | None = None) -> dict[str, Any] | str:
        """
        Fetch daily aggregated account usage statistics synchronously.

        Args:
            params: Optional parameters for formatting the daily usage data.

        Returns:
            The daily usage statistics as a JSON dictionary or CSV string.
        """
        q = (params or AccountUsageDailyParams()).model_dump(exclude_none=True)
        response = self._http.get("/account/usage/daily", params=q)
        return self._parse_response(response)

    async def aget_profile(self) -> dict[str, Any]:
        """
        Retrieve the account profile information asynchronously.

        Returns:
            A dictionary containing profile details.
        """
        response = await self._http.aget("/account/profile")
        return response.json()

    async def aget_balance(self) -> dict[str, Any]:
        """
        Check the current account balance asynchronously.

        Returns:
            A dictionary containing balance information.
        """
        response = await self._http.aget("/account/balance")
        return response.json()

    async def aget_key(self) -> dict[str, Any]:
        """
        Retrieve the current API key information asynchronously.

        Returns:
            A dictionary containing API key details.
        """
        response = await self._http.aget("/account/key")
        return response.json()

    async def aget_usage(
        self, params: AccountUsageParams | None = None
    ) -> dict[str, Any] | str:
        """
        Fetch detailed account usage records asynchronously.

        Args:
            params: Optional parameters for filtering and formatting.

        Returns:
            The usage data as a JSON dictionary or CSV string.
        """
        q = (params or AccountUsageParams()).model_dump(exclude_none=True)
        response = await self._http.aget("/account/usage", params=q)
        return self._parse_response(response)

    async def aget_usage_daily(
        self, params: AccountUsageDailyParams | None = None
    ) -> dict[str, Any] | str:
        """
        Fetch daily aggregated account usage statistics asynchronously.

        Args:
            params: Optional parameters for formatting.

        Returns:
            The daily usage statistics as a JSON dictionary or CSV string.
        """
        q = (params or AccountUsageDailyParams()).model_dump(exclude_none=True)
        response = await self._http.aget("/account/usage/daily", params=q)
        return self._parse_response(response)
