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


def test_list_compatible_models_uses_http_client():
    mi = ModelInformation(api_key="key-123")

    data = mi.list_compatible_models()

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
async def test_alist_compatible_models_uses_async_http_client():
    mi = ModelInformation(api_key="key-async")

    data = await mi.alist_compatible_models()

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


def test_extract_model_ids_with_id_field():
    """Extrae IDs cuando los modelos usan el campo 'id'."""
    models_data = [
        {"id": "openai", "name": "OpenAI", "pricing": {}},
        {"id": "claude", "name": "Claude", "pricing": {}},
        {"id": "gemini", "name": "Gemini", "pricing": {}},
    ]

    result = ModelInformation._extract_model_ids(models_data)

    assert result == ["openai", "claude", "gemini"]


def test_extract_model_ids_with_model_field():
    """Extrae IDs cuando los modelos usan el campo 'model'."""
    models_data = [
        {"model": "flux", "type": "image"},
        {"model": "gptimage", "type": "image"},
    ]

    result = ModelInformation._extract_model_ids(models_data)

    assert result == ["flux", "gptimage"]


def test_extract_model_ids_with_name_field():
    """Extrae IDs cuando los modelos usan el campo 'name'."""
    models_data = [
        {"name": "veo", "capabilities": ["video"]},
        {"name": "seedance", "capabilities": ["image"]},
    ]

    result = ModelInformation._extract_model_ids(models_data)

    assert result == ["veo", "seedance"]


def test_extract_model_ids_with_mixed_fields():
    """Extrae IDs con prioridad: id > model > name."""
    models_data = [
        {"id": "openai", "model": "gpt-4", "name": "GPT-4"},  # Usa 'id'
        {"model": "claude", "name": "Claude"},  # Usa 'model'
        {"name": "gemini"},  # Usa 'name'
    ]

    result = ModelInformation._extract_model_ids(models_data)

    assert result == ["openai", "claude", "gemini"]


def test_extract_model_ids_empty_list():
    """Retorna lista vacía cuando se pasa una lista vacía."""
    result = ModelInformation._extract_model_ids([])

    assert result == []


def test_extract_model_ids_not_a_list():
    """Retorna lista vacía cuando el input no es una lista."""
    result = ModelInformation._extract_model_ids({"not": "a list"})

    assert result == []


def test_extract_model_ids_invalid_items():
    """Ignora items que no son dicts o que no tienen campos de ID válidos."""
    models_data = [
        {"id": "openai"},  # Válido
        "not-a-dict",  # Inválido (string)
        {"no_id_field": "value"},  # Inválido (sin campo de ID)
        {"id": 123},  # Inválido (ID no es string)
        {"model": "claude"},  # Válido
        None,  # Inválido
    ]

    result = ModelInformation._extract_model_ids(models_data)

    assert result == ["openai", "claude"]


def test_get_available_models_success():
    """Obtiene modelos de texto e imagen exitosamente."""
    mi = ModelInformation(api_key="test-key")

    def custom_get(path: str):
        mi._http.calls.append(("get", path))
        if path == "/text/models":
            data = [
                {"id": "openai", "pricing": {}},
                {"id": "claude", "pricing": {}},
                {"id": "mistral", "pricing": {}},
            ]
        elif path == "/image/models":
            data = [
                {"model": "flux", "type": "image"},
                {"model": "gptimage", "type": "image"},
            ]
        else:
            data = {"path": path}
        return DummyResponse(data)

    mi._http.get = custom_get

    result = mi.get_available_models()

    assert result == {
        "text": ["openai", "claude", "mistral"],
        "image": ["flux", "gptimage"],
    }
    assert ("get", "/text/models") in mi._http.calls
    assert ("get", "/image/models") in mi._http.calls


def test_get_available_models_empty_responses():
    """Maneja respuestas vacías correctamente."""
    mi = ModelInformation(api_key="test-key")

    def custom_get(path: str):
        mi._http.calls.append(("get", path))
        return DummyResponse([])

    mi._http.get = custom_get

    result = mi.get_available_models()

    assert result == {"text": [], "image": []}


def test_get_available_models_text_endpoint_fails():
    """Retorna lista vacía para texto si el endpoint falla."""
    mi = ModelInformation(api_key="test-key")

    def custom_get(path: str):
        mi._http.calls.append(("get", path))
        if path == "/text/models":
            raise Exception("Network error")
        elif path == "/image/models":
            return DummyResponse([{"model": "flux"}])
        return DummyResponse([])

    mi._http.get = custom_get

    result = mi.get_available_models()

    assert result == {"text": [], "image": ["flux"]}


def test_get_available_models_image_endpoint_fails():
    """Retorna lista vacía para imagen si el endpoint falla."""
    mi = ModelInformation(api_key="test-key")

    def custom_get(path: str):
        mi._http.calls.append(("get", path))
        if path == "/text/models":
            return DummyResponse([{"id": "openai"}])
        elif path == "/image/models":
            raise Exception("Network error")
        return DummyResponse([])

    mi._http.get = custom_get

    result = mi.get_available_models()

    assert result == {"text": ["openai"], "image": []}


def test_get_available_models_both_endpoints_fail():
    """Retorna listas vacías si ambos endpoints fallan."""
    mi = ModelInformation(api_key="test-key")

    def custom_get(path: str):
        mi._http.calls.append(("get", path))
        raise Exception("Network error")

    mi._http.get = custom_get

    result = mi.get_available_models()

    assert result == {"text": [], "image": []}


def test_get_available_models_invalid_data_format():
    """Maneja respuestas con formato inválido (no lista)."""
    mi = ModelInformation(api_key="test-key")

    def custom_get(path: str):
        mi._http.calls.append(("get", path))
        return DummyResponse({"error": "invalid format"})

    mi._http.get = custom_get

    result = mi.get_available_models()

    assert result == {"text": [], "image": []}


@pytest.mark.asyncio
async def test_aget_available_models_success():
    """Obtiene modelos de texto e imagen exitosamente (async)."""
    mi = ModelInformation(api_key="test-key")

    async def custom_aget(path: str):
        mi._http.calls.append(("aget", path))
        if path == "/text/models":
            data = [
                {"id": "openai"},
                {"id": "claude"},
                {"id": "gemini"},
            ]
        elif path == "/image/models":
            data = [
                {"model": "flux"},
                {"model": "veo"},
                {"model": "seedance"},
            ]
        else:
            data = {"path": path}
        return DummyResponse(data)

    mi._http.aget = custom_aget

    result = await mi.aget_available_models()

    assert result == {
        "text": ["openai", "claude", "gemini"],
        "image": ["flux", "veo", "seedance"],
    }
    assert ("aget", "/text/models") in mi._http.calls
    assert ("aget", "/image/models") in mi._http.calls


@pytest.mark.asyncio
async def test_aget_available_models_empty_responses():
    """Maneja respuestas vacías correctamente (async)."""
    mi = ModelInformation(api_key="test-key")

    async def custom_aget(path: str):
        mi._http.calls.append(("aget", path))
        return DummyResponse([])

    mi._http.aget = custom_aget

    result = await mi.aget_available_models()

    assert result == {"text": [], "image": []}


@pytest.mark.asyncio
async def test_aget_available_models_text_endpoint_fails():
    """Retorna lista vacía para texto si el endpoint falla (async)."""
    mi = ModelInformation(api_key="test-key")

    async def custom_aget(path: str):
        mi._http.calls.append(("aget", path))
        if path == "/text/models":
            raise Exception("Async network error")
        elif path == "/image/models":
            return DummyResponse([{"model": "gptimage"}])
        return DummyResponse([])

    mi._http.aget = custom_aget

    result = await mi.aget_available_models()

    assert result == {"text": [], "image": ["gptimage"]}


@pytest.mark.asyncio
async def test_aget_available_models_image_endpoint_fails():
    """Retorna lista vacía para imagen si el endpoint falla (async)."""
    mi = ModelInformation(api_key="test-key")

    async def custom_aget(path: str):
        mi._http.calls.append(("aget", path))
        if path == "/text/models":
            return DummyResponse([{"id": "mistral"}])
        elif path == "/image/models":
            raise Exception("Async network error")
        return DummyResponse([])

    mi._http.aget = custom_aget

    result = await mi.aget_available_models()

    assert result == {"text": ["mistral"], "image": []}


@pytest.mark.asyncio
async def test_aget_available_models_both_endpoints_fail():
    """Retorna listas vacías si ambos endpoints fallan (async)."""
    mi = ModelInformation(api_key="test-key")

    async def custom_aget(path: str):
        mi._http.calls.append(("aget", path))
        raise Exception("Async network error")

    mi._http.aget = custom_aget

    result = await mi.aget_available_models()

    assert result == {"text": [], "image": []}


@pytest.mark.asyncio
async def test_aget_available_models_mixed_valid_invalid_data():
    """Maneja mezcla de datos válidos e inválidos (async)."""
    mi = ModelInformation(api_key="test-key")

    async def custom_aget(path: str):
        mi._http.calls.append(("aget", path))
        if path == "/text/models":
            data = [
                {"id": "valid-model-1"},
                "invalid-string",
                {"id": "valid-model-2"},
                {"no_id": "invalid"},
            ]
        elif path == "/image/models":
            data = [
                {"model": "image-model-1"},
                {"model": 123},
                {"name": "image-model-2"},
            ]
        else:
            data = []
        return DummyResponse(data)

    mi._http.aget = custom_aget

    result = await mi.aget_available_models()

    assert result == {
        "text": ["valid-model-1", "valid-model-2"],
        "image": ["image-model-1", "image-model-2"],
    }
