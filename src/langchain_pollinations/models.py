from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_pollinations._auth import AuthConfig
from langchain_pollinations._client import HttpConfig, PollinationsHttpClient

DEFAULT_BASE_URL = "https://gen.pollinations.ai"


@dataclass(slots=True)
class ModelInformation:
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

    def list_v1_models(self) -> dict[str, Any]:
        return self._http.get("/v1/models").json()

    def list_text_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        return self._http.get("/text/models").json()

    def list_image_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        return self._http.get("/image/models").json()

    async def alist_v1_models(self) -> dict[str, Any]:
        return (await self._http.aget("/v1/models")).json()

    async def alist_text_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        return (await self._http.aget("/text/models")).json()

    async def alist_image_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        return (await self._http.aget("/image/models")).json()
