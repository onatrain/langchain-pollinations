from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_pollinations.models import DEFAULT_BASE_URL, ModelInformation


def _make_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


SAMPLE_TEXT_MODELS = [
    {"id": "openai", "name": "OpenAI GPT-4"},
    {"id": "mistral", "name": "Mistral"},
    {"model": "llama"},
    {"name": "claude"},
]

SAMPLE_IMAGE_MODELS = [
    {"id": "flux"},
    {"id": "turbo"},
]

SAMPLE_AUDIO_MODELS = [
    {"id": "openai-audio"},
    {"id": "elevenmusic"},
]

SAMPLE_COMPATIBLE_MODELS = {"object": "list", "data": [{"id": "gpt-4"}]}


@pytest.fixture
def mock_http():
    http = MagicMock()
    http.get.return_value = _make_response([])
    http.aget = AsyncMock(return_value=_make_response([]))
    return http


@pytest.fixture
def model_info(mock_http):
    with (
        patch("langchain_pollinations.models.AuthConfig.from_env_or_value") as mock_auth,
        patch("langchain_pollinations.models.PollinationsHttpClient", return_value=mock_http),
    ):
        mock_auth.return_value = MagicMock(api_key="test-key")
        mi = ModelInformation(api_key="test-key")
    # _http already holds mock_http; reassign explicitly for test clarity
    object.__setattr__(mi, "_http", mock_http)
    return mi


class TestExtractModelIds:

    def test_empty_list_returns_empty(self):
        assert ModelInformation._extract_model_ids([]) == []

    def test_not_a_list_returns_empty(self):
        assert ModelInformation._extract_model_ids({}) == []
        assert ModelInformation._extract_model_ids(None) == []
        assert ModelInformation._extract_model_ids("string") == []
        assert ModelInformation._extract_model_ids(42) == []

    def test_extracts_id_field(self):
        data = [{"id": "flux"}, {"id": "turbo"}]
        assert ModelInformation._extract_model_ids(data) == ["flux", "turbo"]

    def test_falls_back_to_model_field(self):
        data = [{"model": "llama"}]
        assert ModelInformation._extract_model_ids(data) == ["llama"]

    def test_falls_back_to_name_field(self):
        data = [{"name": "claude"}]
        assert ModelInformation._extract_model_ids(data) == ["claude"]

    def test_id_takes_priority_over_model_and_name(self):
        data = [{"id": "primary", "model": "secondary", "name": "tertiary"}]
        assert ModelInformation._extract_model_ids(data) == ["primary"]

    def test_model_takes_priority_over_name(self):
        data = [{"model": "secondary", "name": "tertiary"}]
        assert ModelInformation._extract_model_ids(data) == ["secondary"]

    def test_skips_non_dict_items(self):
        data = [{"id": "ok"}, "string", 42, None, {"id": "also_ok"}]
        assert ModelInformation._extract_model_ids(data) == ["ok", "also_ok"]

    def test_skips_items_with_no_known_keys(self):
        data = [{"unknown_key": "value"}, {"id": "valid"}]
        assert ModelInformation._extract_model_ids(data) == ["valid"]

    def test_skips_non_string_id_values(self):
        data = [{"id": 123}, {"id": None}, {"id": "valid"}]
        assert ModelInformation._extract_model_ids(data) == ["valid"]

    def test_full_mixed_sample(self):
        result = ModelInformation._extract_model_ids(SAMPLE_TEXT_MODELS)
        assert result == ["openai", "mistral", "llama", "claude"]


class TestModelInformationInit:

    def _make(self, **kwargs):
        with (
            patch("langchain_pollinations.models.AuthConfig.from_env_or_value") as mock_auth,
            patch("langchain_pollinations.models.PollinationsHttpClient") as mock_cls,
        ):
            mock_auth.return_value = MagicMock(api_key="key")
            mock_cls.return_value = MagicMock()
            mi = ModelInformation(**kwargs)
            return mi, mock_auth, mock_cls

    def test_default_base_url(self):
        mi, _, _ = self._make(api_key="key")
        assert mi.base_url == DEFAULT_BASE_URL

    def test_custom_base_url(self):
        mi, _, _ = self._make(api_key="key", base_url="https://custom.example.com")
        assert mi.base_url == "https://custom.example.com"

    def test_default_timeout(self):
        mi, _, _ = self._make(api_key="key")
        assert mi.timeout_s == 120.0

    def test_custom_timeout(self):
        mi, _, _ = self._make(api_key="key", timeout_s=30.0)
        assert mi.timeout_s == 30.0

    def test_http_client_is_set(self):
        mi, _, mock_cls = self._make(api_key="key")
        assert mi._http is mock_cls.return_value

    def test_api_key_none_delegates_to_auth(self):
        mi, mock_auth, _ = self._make(api_key=None)
        mock_auth.assert_called_once_with(None)

    def test_api_key_passed_to_auth(self):
        mi, mock_auth, _ = self._make(api_key="explicit-key")
        mock_auth.assert_called_once_with("explicit-key")

    def test_http_client_receives_base_url(self):
        from langchain_pollinations.models import PollinationsHttpClient  # noqa: F401
        with (
            patch("langchain_pollinations.models.AuthConfig.from_env_or_value") as mock_auth,
            patch("langchain_pollinations.models.PollinationsHttpClient") as mock_cls,
            patch("langchain_pollinations.models.HttpConfig") as mock_config_cls,
        ):
            mock_auth.return_value = MagicMock(api_key="key")
            mock_cls.return_value = MagicMock()
            ModelInformation(api_key="key", base_url="https://custom.example.com")
            mock_config_cls.assert_called_once_with(
                base_url="https://custom.example.com", timeout_s=120.0
            )


class TestSyncListMethods:

    def test_list_text_models_path_and_return(self, model_info, mock_http):
        mock_http.get.return_value = _make_response(SAMPLE_TEXT_MODELS)
        result = model_info.list_text_models()
        mock_http.get.assert_called_once_with("/text/models")
        assert result == SAMPLE_TEXT_MODELS

    def test_list_image_models_path_and_return(self, model_info, mock_http):
        mock_http.get.return_value = _make_response(SAMPLE_IMAGE_MODELS)
        result = model_info.list_image_models()
        mock_http.get.assert_called_once_with("/image/models")
        assert result == SAMPLE_IMAGE_MODELS

    def test_list_audio_models_path_and_return(self, model_info, mock_http):
        mock_http.get.return_value = _make_response(SAMPLE_AUDIO_MODELS)
        result = model_info.list_audio_models()
        mock_http.get.assert_called_once_with("/audio/models")
        assert result == SAMPLE_AUDIO_MODELS

    def test_list_compatible_models_path_and_return(self, model_info, mock_http):
        mock_http.get.return_value = _make_response(SAMPLE_COMPATIBLE_MODELS)
        result = model_info.list_compatible_models()
        mock_http.get.assert_called_once_with("/v1/models")
        assert result == SAMPLE_COMPATIBLE_MODELS

    def test_list_text_models_returns_dict_payload(self, model_info, mock_http):
        payload = {"models": ["a", "b"]}
        mock_http.get.return_value = _make_response(payload)
        assert model_info.list_text_models() == payload

    def test_list_audio_models_returns_empty_list(self, model_info, mock_http):
        mock_http.get.return_value = _make_response([])
        assert model_info.list_audio_models() == []

    def test_list_image_models_returns_empty_list(self, model_info, mock_http):
        mock_http.get.return_value = _make_response([])
        assert model_info.list_image_models() == []

    def test_list_audio_models_propagates_http_exception(self, model_info, mock_http):
        mock_http.get.side_effect = RuntimeError("connection refused")
        with pytest.raises(RuntimeError, match="connection refused"):
            model_info.list_audio_models()

    def test_list_text_models_propagates_http_exception(self, model_info, mock_http):
        mock_http.get.side_effect = ConnectionError("timeout")
        with pytest.raises(ConnectionError):
            model_info.list_text_models()


class TestAsyncListMethods:
    pytestmark = pytest.mark.asyncio

    async def test_alist_text_models_path_and_return(self, model_info, mock_http):
        mock_http.aget = AsyncMock(return_value=_make_response(SAMPLE_TEXT_MODELS))
        result = await model_info.alist_text_models()
        mock_http.aget.assert_called_once_with("/text/models")
        assert result == SAMPLE_TEXT_MODELS

    async def test_alist_image_models_path_and_return(self, model_info, mock_http):
        mock_http.aget = AsyncMock(return_value=_make_response(SAMPLE_IMAGE_MODELS))
        result = await model_info.alist_image_models()
        mock_http.aget.assert_called_once_with("/image/models")
        assert result == SAMPLE_IMAGE_MODELS

    async def test_alist_audio_models_path_and_return(self, model_info, mock_http):
        mock_http.aget = AsyncMock(return_value=_make_response(SAMPLE_AUDIO_MODELS))
        result = await model_info.alist_audio_models()
        mock_http.aget.assert_called_once_with("/audio/models")
        assert result == SAMPLE_AUDIO_MODELS

    async def test_alist_compatible_models_path_and_return(self, model_info, mock_http):
        mock_http.aget = AsyncMock(return_value=_make_response(SAMPLE_COMPATIBLE_MODELS))
        result = await model_info.alist_compatible_models()
        mock_http.aget.assert_called_once_with("/v1/models")
        assert result == SAMPLE_COMPATIBLE_MODELS

    async def test_alist_audio_models_returns_empty_list(self, model_info, mock_http):
        mock_http.aget = AsyncMock(return_value=_make_response([]))
        assert await model_info.alist_audio_models() == []

    async def test_alist_audio_models_propagates_exception(self, model_info, mock_http):
        mock_http.aget = AsyncMock(side_effect=RuntimeError("network error"))
        with pytest.raises(RuntimeError, match="network error"):
            await model_info.alist_audio_models()

    async def test_alist_text_models_propagates_exception(self, model_info, mock_http):
        mock_http.aget = AsyncMock(side_effect=ConnectionError("timeout"))
        with pytest.raises(ConnectionError):
            await model_info.alist_text_models()


class TestGetAvailableModels:

    def _side_effect_all_ok(self, path):
        mapping = {
            "/text/models": SAMPLE_TEXT_MODELS,
            "/image/models": SAMPLE_IMAGE_MODELS,
            "/audio/models": SAMPLE_AUDIO_MODELS,
        }
        return _make_response(mapping.get(path, []))

    def test_result_has_exactly_three_keys(self, model_info, mock_http):
        mock_http.get.return_value = _make_response([])
        assert set(model_info.get_available_models().keys()) == {"text", "image", "audio"}

    def test_all_catalogs_populated(self, model_info, mock_http):
        mock_http.get.side_effect = self._side_effect_all_ok
        result = model_info.get_available_models()
        assert result["text"] == ["openai", "mistral", "llama", "claude"]
        assert result["image"] == ["flux", "turbo"]
        assert result["audio"] == ["openai-audio", "elevenmusic"]

    def test_text_failure_yields_empty_text(self, model_info, mock_http):
        def _side(path):
            if path == "/text/models":
                raise RuntimeError("text down")
            return _make_response(
                SAMPLE_IMAGE_MODELS if path == "/image/models" else SAMPLE_AUDIO_MODELS
            )
        mock_http.get.side_effect = _side
        result = model_info.get_available_models()
        assert result["text"] == []
        assert result["image"] == ["flux", "turbo"]
        assert result["audio"] == ["openai-audio", "elevenmusic"]

    def test_image_failure_yields_empty_image(self, model_info, mock_http):
        def _side(path):
            if path == "/image/models":
                raise RuntimeError("image down")
            return _make_response(
                SAMPLE_TEXT_MODELS if path == "/text/models" else SAMPLE_AUDIO_MODELS
            )
        mock_http.get.side_effect = _side
        result = model_info.get_available_models()
        assert result["image"] == []
        assert result["text"] == ["openai", "mistral", "llama", "claude"]
        assert result["audio"] == ["openai-audio", "elevenmusic"]

    def test_audio_failure_yields_empty_audio(self, model_info, mock_http):
        def _side(path):
            if path == "/audio/models":
                raise RuntimeError("audio down")
            return _make_response(
                SAMPLE_TEXT_MODELS if path == "/text/models" else SAMPLE_IMAGE_MODELS
            )
        mock_http.get.side_effect = _side
        result = model_info.get_available_models()
        assert result["audio"] == []
        assert result["text"] == ["openai", "mistral", "llama", "claude"]
        assert result["image"] == ["flux", "turbo"]

    def test_all_endpoints_fail_returns_three_empty_lists(self, model_info, mock_http):
        mock_http.get.side_effect = RuntimeError("all down")
        assert model_info.get_available_models() == {"text": [], "image": [], "audio": []}

    def test_does_not_propagate_any_exception(self, model_info, mock_http):
        mock_http.get.side_effect = Exception("unexpected")
        result = model_info.get_available_models()
        assert isinstance(result, dict)

    def test_unexpected_dict_payload_yields_empty_lists(self, model_info, mock_http):
        mock_http.get.return_value = _make_response({"unexpected": "shape"})
        assert model_info.get_available_models() == {"text": [], "image": [], "audio": []}

    def test_value_types_are_lists(self, model_info, mock_http):
        mock_http.get.side_effect = self._side_effect_all_ok
        result = model_info.get_available_models()
        for key in ("text", "image", "audio"):
            assert isinstance(result[key], list)

    def test_each_endpoint_called_exactly_once(self, model_info, mock_http):
        mock_http.get.side_effect = self._side_effect_all_ok
        model_info.get_available_models()
        paths_called = [call.args[0] for call in mock_http.get.call_args_list]
        assert paths_called.count("/text/models") == 1
        assert paths_called.count("/image/models") == 1
        assert paths_called.count("/audio/models") == 1


class TestAgetAvailableModels:
    pytestmark = pytest.mark.asyncio

    async def _side_effect_all_ok(self, path):
        mapping = {
            "/text/models": SAMPLE_TEXT_MODELS,
            "/image/models": SAMPLE_IMAGE_MODELS,
            "/audio/models": SAMPLE_AUDIO_MODELS,
        }
        return _make_response(mapping.get(path, []))

    async def test_result_has_exactly_three_keys(self, model_info, mock_http):
        mock_http.aget = AsyncMock(return_value=_make_response([]))
        result = await model_info.aget_available_models()
        assert set(result.keys()) == {"text", "image", "audio"}

    async def test_all_catalogs_populated(self, model_info, mock_http):
        mock_http.aget.side_effect = self._side_effect_all_ok
        result = await model_info.aget_available_models()
        assert result["text"] == ["openai", "mistral", "llama", "claude"]
        assert result["image"] == ["flux", "turbo"]
        assert result["audio"] == ["openai-audio", "elevenmusic"]

    async def test_text_failure_yields_empty_text(self, model_info, mock_http):
        async def _side(path):
            if path == "/text/models":
                raise RuntimeError("text down")
            return _make_response(
                SAMPLE_IMAGE_MODELS if path == "/image/models" else SAMPLE_AUDIO_MODELS
            )
        mock_http.aget.side_effect = _side
        result = await model_info.aget_available_models()
        assert result["text"] == []
        assert result["image"] == ["flux", "turbo"]
        assert result["audio"] == ["openai-audio", "elevenmusic"]

    async def test_image_failure_yields_empty_image(self, model_info, mock_http):
        async def _side(path):
            if path == "/image/models":
                raise RuntimeError("image down")
            return _make_response(
                SAMPLE_TEXT_MODELS if path == "/text/models" else SAMPLE_AUDIO_MODELS
            )
        mock_http.aget.side_effect = _side
        result = await model_info.aget_available_models()
        assert result["image"] == []
        assert result["text"] == ["openai", "mistral", "llama", "claude"]
        assert result["audio"] == ["openai-audio", "elevenmusic"]

    async def test_audio_failure_yields_empty_audio(self, model_info, mock_http):
        async def _side(path):
            if path == "/audio/models":
                raise RuntimeError("audio down")
            return _make_response(
                SAMPLE_TEXT_MODELS if path == "/text/models" else SAMPLE_IMAGE_MODELS
            )
        mock_http.aget.side_effect = _side
        result = await model_info.aget_available_models()
        assert result["audio"] == []
        assert result["text"] == ["openai", "mistral", "llama", "claude"]
        assert result["image"] == ["flux", "turbo"]

    async def test_all_endpoints_fail_returns_three_empty_lists(self, model_info, mock_http):
        mock_http.aget = AsyncMock(side_effect=RuntimeError("all down"))
        assert await model_info.aget_available_models() == {"text": [], "image": [], "audio": []}

    async def test_does_not_propagate_any_exception(self, model_info, mock_http):
        mock_http.aget = AsyncMock(side_effect=Exception("unexpected"))
        result = await model_info.aget_available_models()
        assert isinstance(result, dict)

    async def test_unexpected_dict_payload_yields_empty_lists(self, model_info, mock_http):
        mock_http.aget = AsyncMock(return_value=_make_response({"unexpected": "shape"}))
        assert await model_info.aget_available_models() == {"text": [], "image": [], "audio": []}

    async def test_value_types_are_lists(self, model_info, mock_http):
        mock_http.aget.side_effect = self._side_effect_all_ok
        result = await model_info.aget_available_models()
        for key in ("text", "image", "audio"):
            assert isinstance(result[key], list)

    async def test_each_endpoint_called_exactly_once(self, model_info, mock_http):
        mock_http.aget.side_effect = self._side_effect_all_ok
        await model_info.aget_available_models()
        paths_called = [call.args[0] for call in mock_http.aget.call_args_list]
        assert paths_called.count("/text/models") == 1
        assert paths_called.count("/image/models") == 1
        assert paths_called.count("/audio/models") == 1
