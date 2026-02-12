from dataclasses import dataclass
from typing import Any

import pytest

from langchain_pollinations.models import ModelInformation, DEFAULT_BASE_URL


@dataclass
class DummyAuth:
    api_key: str


class DummyResponse:
    def __init__(self, data: Any):
        self._data = data

    def json(self) -> Any:
        return self._data


class DummyHttpClient:
    def __init__(self, *, config, api_key: str):
        self.config = config
        self.api_key = api_key
        self.calls: list[tuple[str, str]] = []

    def get(self, path: str):
        self.calls.append(("get", path))
        return DummyResponse({"path": path})

    async def aget(self, path: str):
        self.calls.append(("aget", path))
        return DummyResponse({"path": path})


@pytest.fixture(autouse=True)
def patch_auth_and_http(monkeypatch):
    # Forzar AuthConfig a devolver un DummyAuth sin tocar el entorno.
    monkeypatch.setattr(
        "langchain_pollinations._auth.AuthConfig.from_env_or_value",
        staticmethod(lambda api_key: DummyAuth(api_key=api_key or "dummy")),
    )
    # Reemplazar el cliente HTTP real por uno dummy que no hace red.
    monkeypatch.setattr(
        "langchain_pollinations.models.PollinationsHttpClient",
        DummyHttpClient,
    )
    yield


def test_model_information_initializes_with_defaults():
    mi = ModelInformation()

    assert mi.base_url == DEFAULT_BASE_URL
    assert mi.timeout_s == 120.0
    # _http es un DummyHttpClient según el patch.
    assert isinstance(mi._http, DummyHttpClient)
    assert mi._http.api_key == "dummy"


def test_list_v1_models_uses_http_client():
    mi = ModelInformation(api_key="key-123")

    data = mi.list_v1_models()

    assert data == {"path": "/v1/models"}
    assert mi._http.calls == [("get", "/v1/models")]


def test_list_text_models_uses_http_client():
    mi = ModelInformation(api_key="key-123")

    data = mi.list_text_models()

    assert data == {"path": "/text/models"}
    assert mi._http.calls == [("get", "/text/models")]


def test_list_image_models_uses_http_client():
    mi = ModelInformation(api_key="key-123")

    data = mi.list_image_models()

    assert data == {"path": "/image/models"}
    assert mi._http.calls == [("get", "/image/models")]


@pytest.mark.asyncio
async def test_alist_v1_models_uses_async_http_client():
    mi = ModelInformation(api_key="key-async")

    data = await mi.alist_v1_models()

    assert data == {"path": "/v1/models"}
    assert mi._http.calls == [("aget", "/v1/models")]


@pytest.mark.asyncio
async def test_alist_text_models_uses_async_http_client():
    mi = ModelInformation(api_key="key-async")

    data = await mi.alist_text_models()

    assert data == {"path": "/text/models"}
    assert mi._http.calls == [("aget", "/text/models")]


@pytest.mark.asyncio
async def test_alist_image_models_uses_async_http_client():
    mi = ModelInformation(api_key="key-async")

    data = await mi.alist_image_models()

    assert data == {"path": "/image/models"}
    assert mi._http.calls == [("aget", "/image/models")]
