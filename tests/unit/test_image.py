from pathlib import Path
from typing import Any
import threading
import warnings as _warnings_module
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

import langchain_pollinations.image as image_module
from langchain_pollinations.image import (
    ImagePromptParams,
    ImagePollinations,
    DEFAULT_BASE_URL,
    _FALLBACK_IMAGE_MODEL_IDS,
    _load_image_model_ids,
)


@pytest.fixture
def api_key_from_env(monkeypatch) -> str:
    """
    Lee POLLINATIONS_API_KEY desde .env si existe, y lo inyecta en el entorno.
    Si no existe o no define la variable, usa un valor por defecto para tests.
    """
    env_path = Path(".env")
    api_key = "test_api_key_from_env"
    env_var_name = "POLLINATIONS_API_KEY"

    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == env_var_name:
                api_key = value.strip().strip("'").strip('"')
                break

    monkeypatch.setenv(env_var_name, api_key)
    return api_key


@dataclass
class DummyAuth:
    api_key: str


class DummyResponse:
    def __init__(self, content: bytes, headers: dict[str, str] | None = None):
        self.content = content
        self.headers = headers or {"content-type": "image/png"}
        self.status_code = 200
        self.url = "https://example.com/image/test"


class DummyHttpClient:
    def __init__(self, *, config, api_key: str):
        self.config = config
        self.api_key = api_key
        self.calls: list[dict[str, Any]] = []

    def get(self, path: str, params: dict[str, Any] | None = None):
        self.calls.append({"method": "get", "path": path, "params": params})
        return DummyResponse(b"fake_image_data")

    async def aget(self, path: str, params: dict[str, Any] | None = None):
        self.calls.append({"method": "aget", "path": path, "params": params})
        return DummyResponse(b"fake_async_image_data")


@pytest.fixture(autouse=True)
def patch_auth_and_http(monkeypatch):
    """Mock automático de AuthConfig y PollinationsHttpClient para evitar llamadas reales."""
    monkeypatch.setattr(
        "langchain_pollinations._auth.AuthConfig.from_env_or_value",
        staticmethod(lambda api_key: DummyAuth(api_key=api_key or "dummy")),
    )
    monkeypatch.setattr(
        "langchain_pollinations.image.PollinationsHttpClient",
        DummyHttpClient,
    )
    yield


def test_image_prompt_params_default_values():
    params = ImagePromptParams()

    assert params.model == "zimage"
    assert params.width == 1024
    assert params.height == 1024
    assert params.seed == 0
    assert params.enhance is False
    assert params.negative_prompt == "worst quality, blurry"
    assert params.safe is False
    assert params.quality == "medium"
    assert params.image is None
    assert params.transparent is False
    assert params.duration is None
    assert params.aspect_ratio is None
    assert params.audio is False


def test_image_prompt_params_custom_values():
    params = ImagePromptParams(
        model="flux",
        width=512,
        height=768,
        seed=42,
        enhance=True,
        negative_prompt="bad quality",
        safe=True,
        quality="high",
        image="https://example.com/ref.jpg",
        transparent=True,
        duration=5,
        aspect_ratio="16:9",
        audio=True,
    )

    assert params.model == "flux"
    assert params.width == 512
    assert params.height == 768
    assert params.seed == 42
    assert params.enhance is True
    assert params.negative_prompt == "bad quality"
    assert params.safe is True
    assert params.quality == "high"
    assert params.image == "https://example.com/ref.jpg"
    assert params.transparent is True
    assert params.duration == 5
    assert params.aspect_ratio == "16:9"
    assert params.audio is True


def test_image_prompt_params_seed_validation():
    # seed debe estar en rango [-1, 2147483647]
    with pytest.raises(ValidationError):
        ImagePromptParams(seed=-2)

    with pytest.raises(ValidationError):
        ImagePromptParams(seed=2_147_483_648)

    # Valores límite válidos
    params_min = ImagePromptParams(seed=-1)
    assert params_min.seed == -1

    params_max = ImagePromptParams(seed=2_147_483_647)
    assert params_max.seed == 2_147_483_647


def test_image_prompt_params_width_height_validation():
    # width y height deben ser >= 0
    with pytest.raises(ValidationError):
        ImagePromptParams(width=-1)

    with pytest.raises(ValidationError):
        ImagePromptParams(height=-1)

    params = ImagePromptParams(width=0, height=0)
    assert params.width == 0
    assert params.height == 0


def test_image_prompt_params_duration_validation():
    # duration debe estar en rango [1, 10] si se especifica
    with pytest.raises(ValidationError):
        ImagePromptParams(duration=0)

    with pytest.raises(ValidationError):
        ImagePromptParams(duration=11)

    params_min = ImagePromptParams(duration=1)
    assert params_min.duration == 1

    params_max = ImagePromptParams(duration=10)
    assert params_max.duration == 10


def test_image_prompt_params_to_query_excludes_none():
    params = ImagePromptParams(
        model="flux",
        width=512,
        image=None,
        duration=None,
    )

    query = params.to_query()

    assert "model" in query
    assert "width" in query
    assert "image" not in query
    assert "duration" not in query


def test_image_prompt_params_to_query_uses_aliases():
    params = ImagePromptParams(aspect_ratio="16:9")

    query = params.to_query()

    # El alias es aspectRatio (camelCase)
    assert "aspectRatio" in query
    assert query["aspectRatio"] == "16:9"
    assert "aspect_ratio" not in query


def test_image_prompt_params_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ImagePromptParams(model="flux", unknown_field="value")  # type: ignore[call-arg]


def test_image_pollinations_initializes_with_defaults():
    img = ImagePollinations()

    assert img.base_url == DEFAULT_BASE_URL
    assert img.timeout_s == 120.0
    assert img.model is None
    assert img.width is None
    assert img.height is None
    assert isinstance(img._http, DummyHttpClient)


def test_image_pollinations_initializes_with_custom_params():
    img = ImagePollinations(
        api_key="custom-key",
        base_url="https://custom.api",
        timeout_s=60.0,
        model="flux",
        width=512,
        height=768,
        seed=99,
        enhance=True,
    )

    assert img.base_url == "https://custom.api"
    assert img.timeout_s == 60.0
    assert img.model == "flux"
    assert img.width == 512
    assert img.height == 768
    assert img.seed == 99
    assert img.enhance is True


def test_image_pollinations_with_params_clones_and_overrides():
    img = ImagePollinations(model="flux", width=512)

    cloned = img.with_params(width=1024, height=768)

    # Original no debe cambiar
    assert img.width == 512
    assert img.height is None

    # Clonado debe tener los overrides
    assert cloned.model == "flux"
    assert cloned.width == 1024
    assert cloned.height == 768


def test_image_pollinations_defaults_dict_returns_configured_params():
    img = ImagePollinations(
        model="flux",
        width=512,
        enhance=True,
        aspect_ratio="16:9",
    )

    defaults = img._defaults_dict()

    assert defaults["model"] == "flux"
    assert defaults["width"] == 512
    assert defaults["enhance"] is True
    assert defaults["aspectRatio"] == "16:9"
    # Campos no configurados no deben aparecer
    assert "height" not in defaults
    assert "seed" not in defaults


def test_image_pollinations_build_query_merges_params():
    img = ImagePollinations(model="flux", width=512)

    query = img._build_query(params={"height": 768}, seed=42)

    assert query["model"] == "flux"
    assert query["width"] == 512
    assert query["height"] == 768
    assert query["seed"] == 42


def test_image_pollinations_build_query_validates_with_schema():
    img = ImagePollinations()

    # seed fuera de rango debe fallar en validación
    with pytest.raises(ValidationError):
        img._build_query(seed=-2)


def test_image_pollinations_generate_response_calls_http_get():
    img = ImagePollinations(model="flux")

    response = img.generate_response("a cat", width=512, height=768)

    assert response.content == b"fake_image_data"
    assert len(img._http.calls) == 1
    call = img._http.calls[0]
    assert call["method"] == "get"
    assert "/image/a%20cat" in call["path"]
    params = call["params"]
    assert params["model"] == "flux"
    assert params["width"] == 512
    assert params["height"] == 768


def test_image_pollinations_generate_response_encodes_prompt():
    img = ImagePollinations()

    response = img.generate_response("hello world / test?")

    call = img._http.calls[0]
    # El prompt debe estar URL-encoded
    assert "/image/hello%20world%20%2F%20test%3F" in call["path"]


@pytest.mark.asyncio
async def test_image_pollinations_agenerate_response_calls_http_aget():
    img = ImagePollinations(model="kontext", seed=123)

    response = await img.agenerate_response("async cat", width=256)

    assert response.content == b"fake_async_image_data"
    assert len(img._http.calls) == 1
    call = img._http.calls[0]
    assert call["method"] == "aget"
    assert "/image/async%20cat" in call["path"]
    params = call["params"]
    assert params["model"] == "kontext"
    assert params["seed"] == 123
    assert params["width"] == 256


def test_image_pollinations_generate_returns_bytes():
    img = ImagePollinations()

    result = img.generate("test prompt")

    assert isinstance(result, bytes)
    assert result == b"fake_image_data"


@pytest.mark.asyncio
async def test_image_pollinations_agenerate_returns_bytes():
    img = ImagePollinations()

    result = await img.agenerate("async test prompt")

    assert isinstance(result, bytes)
    assert result == b"fake_async_image_data"


def test_image_pollinations_generate_with_params_dict():
    img = ImagePollinations()

    result = img.generate("test", params={"model": "flux", "width": 640})

    assert result == b"fake_image_data"
    call = img._http.calls[0]
    params = call["params"]
    assert params["model"] == "flux"
    assert params["width"] == 640


def test_image_pollinations_invoke_delegates_to_generate():
    img = ImagePollinations(model="seedream")

    result = img.invoke("runnable test", width=512)

    assert result == b"fake_image_data"
    call = img._http.calls[0]
    assert "/image/runnable%20test" in call["path"]
    params = call["params"]
    assert params["model"] == "seedream"
    assert params["width"] == 512


@pytest.mark.asyncio
async def test_image_pollinations_ainvoke_delegates_to_agenerate():
    img = ImagePollinations(model="gptimage")

    result = await img.ainvoke("async runnable test", height=1024)

    assert result == b"fake_async_image_data"
    call = img._http.calls[0]
    assert "/image/async%20runnable%20test" in call["path"]
    params = call["params"]
    assert params["model"] == "gptimage"
    assert params["height"] == 1024


def test_image_pollinations_invoke_ignores_config():
    img = ImagePollinations()

    # config no debe afectar el resultado
    result = img.invoke("test", config={"some": "config"})

    assert result == b"fake_image_data"


@pytest.mark.asyncio
async def test_image_pollinations_ainvoke_ignores_config():
    img = ImagePollinations()

    result = await img.ainvoke("test", config={"some": "config"})

    assert result == b"fake_async_image_data"


def test_image_pollinations_generate_with_all_params():
    img = ImagePollinations()

    result = img.generate(
        "complex prompt",
        params={
            "model": "wan",
            "width": 800,
            "height": 600,
            "seed": 444,
            "enhance": True,
            "negative_prompt": "ugly",
            "safe": True,
            "quality": "hd",
            "image": "https://ref.com/img.jpg",
            "transparent": True,
            "duration": 7,
            "aspectRatio": "4:3",
            "audio": True,
        }
    )

    assert result == b"fake_image_data"
    call = img._http.calls[0]
    params = call["params"]
    assert params["model"] == "wan"
    assert params["width"] == 800
    assert params["height"] == 600
    assert params["seed"] == 444
    assert params["enhance"] is True
    assert params["negative_prompt"] == "ugly"
    assert params["safe"] is True
    assert params["quality"] == "hd"
    assert params["image"] == "https://ref.com/img.jpg"
    assert params["transparent"] is True
    assert params["duration"] == 7
    assert params["aspectRatio"] == "4:3"
    assert params["audio"] is True


def test_image_pollinations_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ImagePollinations(model="flux", unknown_param="value")  # type: ignore[call-arg]


def test_image_pollinations_with_params_preserves_api_key():
    img = ImagePollinations(api_key="secret-123", model="flux")

    cloned = img.with_params(width=512)

    # api_key debe preservarse aunque se excluya del dump
    assert cloned.api_key == "secret-123"
    assert cloned._http.api_key == "secret-123"


def test_image_pollinations_generate_merges_instance_and_call_params():
    img = ImagePollinations(model="klein", width=1024, seed=42)

    # Los parámetros de llamada deben sobrescribir los de instancia
    result = img.generate("test", width=512, height=768)

    call = img._http.calls[0]
    params = call["params"]
    assert params["model"] == "klein"  # de instancia
    assert params["width"] == 512      # sobrescrito en llamada
    assert params["height"] == 768     # nuevo en llamada
    assert params["seed"] == 42        # de instancia


@pytest.mark.asyncio
async def test_image_pollinations_agenerate_merges_instance_and_call_params():
    img = ImagePollinations(model="seedance-pro", enhance=True)

    result = await img.agenerate("async test", enhance=False, safe=True)

    call = img._http.calls[0]
    params = call["params"]
    assert params["model"] == "seedance-pro"
    assert params["enhance"] is False  # sobrescrito
    assert params["safe"] is True      # nuevo


@pytest.fixture()
def reset_image_catalog(monkeypatch):
    """
    Restore module-level catalog state to its initial (fallback) values before
    each test that exercises ``_load_image_model_ids`` directly.

    Resets all three module globals that participate in the one-shot loading
    mechanism, including a fresh lock to avoid any contamination from a lock
    object that may have been acquired in a previous test.
    """
    # Restaurar caché al fallback para evitar contaminación entre tests.
    monkeypatch.setattr(image_module, "_image_model_ids_cache", list(_FALLBACK_IMAGE_MODEL_IDS))
    # Marcar como NO cargado para que cada test empiece desde cero.
    monkeypatch.setattr(image_module, "_image_model_ids_loaded", False)
    # Lock fresco: evita que un lock adquirido en un test anterior bloquee al siguiente.
    monkeypatch.setattr(image_module, "_image_model_ids_lock", threading.Lock())


def test_fallback_catalog_is_not_empty():
    """
    Verify that the hardcoded fallback catalog contains at least one model ID.
    An empty fallback would make the cache meaningless before the first API call.
    """
    assert len(_FALLBACK_IMAGE_MODEL_IDS) > 0


def test_fallback_catalog_contains_flux():
    """
    Verify that 'flux' is present in the fallback catalog.
    """
    assert "flux" in _FALLBACK_IMAGE_MODEL_IDS


def test_load_image_model_ids_updates_cache_on_success(reset_image_catalog):
    """
    Cache is replaced with the API response when the call succeeds and the
    response contains at least one model ID.
    """
    api_models = ["flux", "turbo", "gptimage"]
    mock_info = MagicMock()
    mock_info.get_available_models.return_value = {"image": api_models}

    with patch("langchain_pollinations.models.ModelInformation", return_value=mock_info):
        result = _load_image_model_ids(api_key="test_key")

    assert result == api_models
    assert image_module._image_model_ids_cache == api_models


def test_load_image_model_ids_retains_fallback_on_api_failure(reset_image_catalog):
    """
    Cache is not modified when the API call raises an exception; the fallback
    list is preserved so the library remains functional without connectivity.
    """
    with patch(
        "langchain_pollinations.models.ModelInformation",
        side_effect=RuntimeError("network error"),
    ):
        result = _load_image_model_ids(api_key="test_key")

    assert result == list(_FALLBACK_IMAGE_MODEL_IDS)
    assert image_module._image_model_ids_cache == list(_FALLBACK_IMAGE_MODEL_IDS)


def test_load_image_model_ids_skips_empty_api_response(reset_image_catalog):
    """
    Cache is not replaced when the API returns an empty list.
    An empty response is treated as a non-authoritative answer.
    """
    mock_info = MagicMock()
    mock_info.get_available_models.return_value = {"image": []}

    with patch("langchain_pollinations.models.ModelInformation", return_value=mock_info):
        result = _load_image_model_ids(api_key="test_key")

    assert result == list(_FALLBACK_IMAGE_MODEL_IDS)
    assert image_module._image_model_ids_cache == list(_FALLBACK_IMAGE_MODEL_IDS)


def test_load_image_model_ids_marks_loaded_on_success(reset_image_catalog):
    """
    ``_image_model_ids_loaded`` is set to ``True`` after a successful API call,
    activating the one-shot guard for subsequent invocations.
    """
    mock_info = MagicMock()
    mock_info.get_available_models.return_value = {"image": ["flux"]}

    with patch("langchain_pollinations.models.ModelInformation", return_value=mock_info):
        _load_image_model_ids(api_key="test_key")

    assert image_module._image_model_ids_loaded is True


def test_load_image_model_ids_marks_loaded_on_failure(reset_image_catalog):
    """
    ``_image_model_ids_loaded`` is set to ``True`` even when the API call fails.

    This prevents the library from retrying on every instantiation in environments
    without connectivity. A forced refresh via ``force=True`` is the explicit path.
    """
    with patch(
        "langchain_pollinations.models.ModelInformation",
        side_effect=ConnectionError("no network"),
    ):
        _load_image_model_ids(api_key="test_key")

    assert image_module._image_model_ids_loaded is True


def test_load_image_model_ids_one_shot_guard(reset_image_catalog):
    """
    The API is called exactly once; subsequent calls reuse the cache without
    constructing a new ``ModelInformation`` instance.
    """
    mock_info = MagicMock()
    mock_info.get_available_models.return_value = {"image": ["flux", "turbo"]}

    with patch(
        "langchain_pollinations.models.ModelInformation", return_value=mock_info
    ) as mock_cls:
        _load_image_model_ids(api_key="test_key")
        _load_image_model_ids(api_key="test_key")
        _load_image_model_ids(api_key="test_key")

    # Solo una construcción de ModelInformation, independientemente de cuántas
    # veces se llame a _load_image_model_ids sin force=True.
    assert mock_cls.call_count == 1


def test_load_image_model_ids_force_bypasses_guard(reset_image_catalog):
    """
    ``force=True`` causes the API to be called again regardless of the loaded flag,
    and the cache is updated with the fresh response.
    """
    api_models_v1 = ["flux"]
    api_models_v2 = ["flux", "turbo", "gptimage"]

    mock_v1 = MagicMock()
    mock_v1.get_available_models.return_value = {"image": api_models_v1}
    mock_v2 = MagicMock()
    mock_v2.get_available_models.return_value = {"image": api_models_v2}

    with patch(
        "langchain_pollinations.models.ModelInformation",
        side_effect=[mock_v1, mock_v2],
    ) as mock_cls:
        _load_image_model_ids(api_key="test_key")            # primera carga
        result = _load_image_model_ids(api_key="test_key", force=True)  # refresco forzado

    assert mock_cls.call_count == 2
    assert result == api_models_v2
    assert image_module._image_model_ids_cache == api_models_v2


def test_load_image_model_ids_force_skips_empty_response_too(reset_image_catalog):
    """
    Even with ``force=True``, an empty API response does not wipe the cache.
    The existing content (populated by a previous successful call) is retained.
    """
    populated_models = ["flux", "turbo"]
    mock_v1 = MagicMock()
    mock_v1.get_available_models.return_value = {"image": populated_models}
    mock_v2 = MagicMock()
    mock_v2.get_available_models.return_value = {"image": []}

    with patch(
        "langchain_pollinations.models.ModelInformation",
        side_effect=[mock_v1, mock_v2],
    ):
        _load_image_model_ids(api_key="test_key")
        _load_image_model_ids(api_key="test_key", force=True)

    # El caché sigue teniendo el contenido de la primera carga exitosa.
    assert image_module._image_model_ids_cache == populated_models


def test_load_image_model_ids_returns_copy(reset_image_catalog):
    """
    The returned list is an independent copy; mutating it does not affect the
    internal cache, preventing accidental external modification of shared state.
    """
    mock_info = MagicMock()
    mock_info.get_available_models.return_value = {"image": ["flux"]}

    with patch("langchain_pollinations.models.ModelInformation", return_value=mock_info):
        result = _load_image_model_ids(api_key="test_key")

    result.append("injected_model")

    assert "injected_model" not in image_module._image_model_ids_cache


def test_imagepollinations_known_model_no_warning(monkeypatch):
    """
    No catalog warning is emitted when the model ID is present in the cache.
    """
    monkeypatch.setattr(image_module, "_image_model_ids_loaded", True)
    monkeypatch.setattr(image_module, "_image_model_ids_cache", ["flux", "turbo", "gptimage"])

    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always")
        ImagePollinations(api_key="test_key", model="flux")

    # Filtrar exclusivamente los warnings del validador de catálogo.
    catalog_warns = [w for w in caught if "image model catalog" in str(w.message).lower()]
    assert len(catalog_warns) == 0


def test_imagepollinations_unknown_model_emits_warning(monkeypatch):
    """
    A ``UserWarning`` is emitted when the model ID is absent from the cache,
    and the warning message identifies the offending ID.
    """
    monkeypatch.setattr(image_module, "_image_model_ids_loaded", True)
    monkeypatch.setattr(image_module, "_image_model_ids_cache", ["flux", "turbo"])

    with pytest.warns(UserWarning, match="unknown-xyz-model"):
        ImagePollinations(api_key="test_key", model="unknown-xyz-model")


def test_imagepollinations_unknown_model_warning_mentions_catalog_size(monkeypatch):
    """
    The warning message includes the number of models in the loaded catalog,
    giving the user context to interpret the warning.
    """
    catalog = ["flux", "turbo", "gptimage"]
    monkeypatch.setattr(image_module, "_image_model_ids_loaded", True)
    monkeypatch.setattr(image_module, "_image_model_ids_cache", catalog)

    with pytest.warns(UserWarning, match=str(len(catalog))):
        ImagePollinations(api_key="test_key", model="nonexistent-model")


def test_imagepollinations_none_model_no_warning(monkeypatch):
    """
    No warning is emitted when ``model=None``; ``None`` means "use the API
    default" and is always accepted without consulting the catalog.
    """
    monkeypatch.setattr(image_module, "_image_model_ids_loaded", True)
    monkeypatch.setattr(image_module, "_image_model_ids_cache", ["flux", "turbo"])

    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always")
        ImagePollinations(api_key="test_key", model=None)

    catalog_warns = [w for w in caught if "image model catalog" in str(w.message).lower()]
    assert len(catalog_warns) == 0


def test_imagepollinations_warning_emitted_only_once_per_instance(monkeypatch):
    """
    The catalog warning fires exactly once when the instance is created, not on
    every subsequent call to ``generate()`` or ``invoke()``.

    This guards against a regression where ``_validate_model_id`` was mistakenly
    placed in ``ImagePromptParams``, causing it to fire on every ``_build_query()``
    call (i.e., every generation request).
    """
    monkeypatch.setattr(image_module, "_image_model_ids_loaded", True)
    monkeypatch.setattr(image_module, "_image_model_ids_cache", ["flux"])

    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always")
        # Instanciar con un modelo desconocido emite el warning.
        client = ImagePollinations(api_key="test_key", model="unknown-model")
        # Construir la query interna directamente (simula lo que _build_query hace
        # por cada generate()) no debe emitir nuevos warnings de catálogo.
        client._build_query()
        client._build_query()

    catalog_warns = [w for w in caught if "image model catalog" in str(w.message).lower()]
    # Solo el warning de la construcción del ImagePollinations; ninguno de _build_query.
    assert len(catalog_warns) == 1
