from __future__ import annotations

import typing
import warnings
from contextlib import contextmanager
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from pydantic import ValidationError

import langchain_pollinations.tts as tts_module
from langchain_pollinations.tts import (
    DEFAULT_BASE_URL,
    AudioFormat,
    SpeechRequest,
    TTSPollinations,
    VoiceId,
)
from langchain_pollinations._audio_catalog import (
    AudioModelId,
    _FALLBACK_AUDIO_MODEL_IDS,
    _load_audio_model_ids,
    _audio_model_ids_loaded,
)


@pytest.fixture(autouse=True)
def _reset_audio_catalog() -> Generator[None, None, None]:
    """Restore module-level catalog state after each test to prevent pollution."""
    original_cache = list(tts_module._audio_model_ids_cache)
    original_loaded = tts_module._audio_model_ids_loaded
    yield
    tts_module._audio_model_ids_cache = original_cache
    tts_module._audio_model_ids_loaded = original_loaded


@pytest.fixture
def mock_http() -> MagicMock:
    """Return a mock PollinationsHttpClient with pre-wired sync/async methods."""
    m = MagicMock()
    m.post_json = MagicMock()
    m.apost_json = AsyncMock()
    return m


@contextmanager
def _tts_patches(mock_http: MagicMock) -> Generator[None, None, None]:
    """
    Context manager that patches all TTSPollinations construction dependencies.

    Suppresses catalog loading, auth resolution, and HTTP client instantiation
    so that tests can build instances without a live API key or network.
    """
    with (
        patch("langchain_pollinations.tts._load_audio_model_ids"),
        patch("langchain_pollinations.tts.AuthConfig") as mock_auth_cls,
        patch("langchain_pollinations.tts.PollinationsHttpClient", return_value=mock_http),
    ):
        mock_auth_cls.from_env_or_value.return_value.api_key = "test-key"
        yield


def _make_tts(mock_http: MagicMock, **kwargs: Any) -> TTSPollinations:
    """
    Build a TTSPollinations instance bypassing real auth and HTTP client creation.

    Args:
        mock_http: Pre-configured mock HTTP client injected into the instance.
        **kwargs: Extra fields forwarded to TTSPollinations constructor.

    Returns:
        A fully initialised TTSPollinations instance backed by ``mock_http``.
    """
    with _tts_patches(mock_http):
        return TTSPollinations(api_key="test-key", **kwargs)


def _fake_response(content: bytes = b"audio_data") -> MagicMock:
    """
    Return a minimal httpx.Response mock carrying binary audio content.

    Args:
        content: Bytes to expose via the ``.content`` attribute.

    Returns:
        A MagicMock spec'd as httpx.Response with ``.content`` set.
    """
    resp = MagicMock(spec=httpx.Response)
    resp.content = content
    return resp


@pytest.fixture
def tts(mock_http: MagicMock) -> TTSPollinations:
    """
    Return a bare TTSPollinations instance with no request defaults configured.

    Depends on the ``mock_http`` fixture so tests can inspect calls on the
    shared mock after invoking any generation method.
    """
    with _tts_patches(mock_http):
        return TTSPollinations(api_key="test-key")


class TestFallbackList:
    def test_is_list_of_strings(self) -> None:
        assert isinstance(_FALLBACK_AUDIO_MODEL_IDS, list)
        assert all(isinstance(m, str) for m in _FALLBACK_AUDIO_MODEL_IDS)

    def test_contains_openai_audio(self) -> None:
        assert "openai-audio" in _FALLBACK_AUDIO_MODEL_IDS

    def test_contains_tts_1(self) -> None:
        assert "tts-1" in _FALLBACK_AUDIO_MODEL_IDS

    def test_contains_elevenmusic(self) -> None:
        assert "elevenmusic" in _FALLBACK_AUDIO_MODEL_IDS

    def test_is_non_empty(self) -> None:
        assert len(_FALLBACK_AUDIO_MODEL_IDS) > 0


class TestPublicTypes:
    def test_audio_model_id_is_str_alias(self) -> None:
        assert AudioModelId is str

    def test_voice_id_is_str_alias(self) -> None:
        assert VoiceId is str

    def test_audio_format_literal_has_correct_values(self) -> None:
        args = set(typing.get_args(AudioFormat))
        assert args == {"mp3", "opus", "aac", "flac", "wav", "pcm"}

    def test_default_base_url_value(self) -> None:
        assert DEFAULT_BASE_URL == "https://gen.pollinations.ai"


class TestSpeechRequest:
    def test_minimal_valid_request_only_requires_input(self) -> None:
        req = SpeechRequest(input="Hello")
        assert req.input == "Hello"

    def test_defaults_match_api_spec(self) -> None:
        req = SpeechRequest(input="Hello")
        assert req.voice == "alloy"
        assert req.response_format == "mp3"
        assert req.speed == 1.0
        assert req.model is None
        assert req.duration is None
        assert req.instrumental is None

    def test_to_body_returns_dict(self) -> None:
        body = SpeechRequest(input="Test").to_body()
        assert isinstance(body, dict)

    def test_to_body_excludes_none_model(self) -> None:
        body = SpeechRequest(input="Hello").to_body()
        assert "model" not in body

    def test_to_body_excludes_none_duration(self) -> None:
        body = SpeechRequest(input="Hello").to_body()
        assert "duration" not in body

    def test_to_body_excludes_none_instrumental(self) -> None:
        body = SpeechRequest(input="Hello").to_body()
        assert "instrumental" not in body

    def test_to_body_includes_input(self) -> None:
        body = SpeechRequest(input="My text").to_body()
        assert body["input"] == "My text"

    def test_to_body_includes_defaults_for_voice_format_speed(self) -> None:
        body = SpeechRequest(input="Hello").to_body()
        assert body["voice"] == "alloy"
        assert body["response_format"] == "mp3"
        assert body["speed"] == 1.0

    def test_to_body_includes_all_fields_when_set(self) -> None:
        req = SpeechRequest(
            input="Hello",
            model="tts-1",
            voice="rachel",
            response_format="wav",
            speed=1.5,
            duration=30.0,
            instrumental=True,
        )
        body = req.to_body()
        assert body["input"] == "Hello"
        assert body["model"] == "tts-1"
        assert body["voice"] == "rachel"
        assert body["response_format"] == "wav"
        assert body["speed"] == 1.5
        assert body["duration"] == 30.0
        assert body["instrumental"] is True

    def test_to_body_instrumental_false_is_included_not_excluded(self) -> None:
        # False no es None; debe aparecer en el body serializado.
        body = SpeechRequest(input="Hello", instrumental=False).to_body()
        assert "instrumental" in body
        assert body["instrumental"] is False

    def test_to_body_returns_independent_copies_each_call(self) -> None:
        req = SpeechRequest(input="Hello")
        body1 = req.to_body()
        body2 = req.to_body()
        assert body1 == body2
        assert body1 is not body2

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(input="")

    def test_raises_on_input_exceeding_max_length(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(input="x" * 4097)

    def test_accepts_input_at_min_length(self) -> None:
        req = SpeechRequest(input="a")
        assert req.input == "a"

    def test_accepts_input_at_max_length(self) -> None:
        req = SpeechRequest(input="x" * 4096)
        assert len(req.input) == 4096

    def test_raises_on_speed_below_minimum(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(input="Hello", speed=0.24)

    def test_raises_on_speed_above_maximum(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(input="Hello", speed=4.01)

    def test_accepts_speed_at_lower_boundary(self) -> None:
        req = SpeechRequest(input="Hello", speed=0.25)
        assert req.speed == 0.25

    def test_accepts_speed_at_upper_boundary(self) -> None:
        req = SpeechRequest(input="Hello", speed=4.0)
        assert req.speed == 4.0

    def test_raises_on_duration_below_minimum(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(input="Hello", duration=2.9)

    def test_raises_on_duration_above_maximum(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(input="Hello", duration=300.1)

    def test_accepts_duration_at_lower_boundary(self) -> None:
        req = SpeechRequest(input="Hello", duration=3.0)
        assert req.duration == 3.0

    def test_accepts_duration_at_upper_boundary(self) -> None:
        req = SpeechRequest(input="Hello", duration=300.0)
        assert req.duration == 300.0

    def test_raises_on_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(input="Hello", unexpected_field="value")

    @pytest.mark.parametrize("fmt", ["mp3", "opus", "aac", "flac", "wav", "pcm"])
    def test_all_valid_audio_formats_accepted(self, fmt: str) -> None:
        req = SpeechRequest(input="Hello", response_format=fmt)  # type: ignore[arg-type]
        assert req.response_format == fmt

    def test_raises_on_invalid_response_format(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(input="Hello", response_format="ogg")  # type: ignore[arg-type]

    def test_elevenmusic_params_present_in_body(self) -> None:
        req = SpeechRequest(
            input="A soft jazz tune",
            model="elevenmusic",
            duration=60.0,
            instrumental=True,
        )
        body = req.to_body()
        assert body["model"] == "elevenmusic"
        assert body["duration"] == 60.0
        assert body["instrumental"] is True


class TestTTSPollinationsInit:
    def test_calls_load_audio_model_ids_with_api_key(self, mock_http: MagicMock) -> None:
        with (
            patch("langchain_pollinations.tts._load_audio_model_ids") as mock_load,
            patch("langchain_pollinations.tts.AuthConfig") as mock_auth_cls,
            patch("langchain_pollinations.tts.PollinationsHttpClient", return_value=mock_http),
        ):
            mock_auth_cls.from_env_or_value.return_value.api_key = "sk-abc"
            TTSPollinations(api_key="sk-abc")
        mock_load.assert_called_once_with("sk-abc")

    def test_calls_auth_config_with_api_key(self, mock_http: MagicMock) -> None:
        with (
            patch("langchain_pollinations.tts._load_audio_model_ids"),
            patch("langchain_pollinations.tts.AuthConfig") as mock_auth_cls,
            patch("langchain_pollinations.tts.PollinationsHttpClient", return_value=mock_http),
        ):
            mock_auth_cls.from_env_or_value.return_value.api_key = "sk-abc"
            TTSPollinations(api_key="sk-abc")
        mock_auth_cls.from_env_or_value.assert_called_once_with("sk-abc")

    def test_creates_http_client_with_default_base_url(self, mock_http: MagicMock) -> None:
        with (
            patch("langchain_pollinations.tts._load_audio_model_ids"),
            patch("langchain_pollinations.tts.AuthConfig") as mock_auth_cls,
            patch("langchain_pollinations.tts.PollinationsHttpClient") as mock_http_cls,
        ):
            mock_auth_cls.from_env_or_value.return_value.api_key = "sk"
            mock_http_cls.return_value = mock_http
            TTSPollinations(api_key="sk")
        _, kwargs = mock_http_cls.call_args
        assert kwargs["config"].base_url == DEFAULT_BASE_URL

    def test_creates_http_client_with_custom_base_url(self, mock_http: MagicMock) -> None:
        with (
            patch("langchain_pollinations.tts._load_audio_model_ids"),
            patch("langchain_pollinations.tts.AuthConfig") as mock_auth_cls,
            patch("langchain_pollinations.tts.PollinationsHttpClient") as mock_http_cls,
        ):
            mock_auth_cls.from_env_or_value.return_value.api_key = "sk"
            mock_http_cls.return_value = mock_http
            TTSPollinations(api_key="sk", base_url="https://custom.example.com")
        _, kwargs = mock_http_cls.call_args
        assert kwargs["config"].base_url == "https://custom.example.com"

    def test_creates_http_client_with_custom_timeout(self, mock_http: MagicMock) -> None:
        with (
            patch("langchain_pollinations.tts._load_audio_model_ids"),
            patch("langchain_pollinations.tts.AuthConfig") as mock_auth_cls,
            patch("langchain_pollinations.tts.PollinationsHttpClient") as mock_http_cls,
        ):
            mock_auth_cls.from_env_or_value.return_value.api_key = "sk"
            mock_http_cls.return_value = mock_http
            TTSPollinations(api_key="sk", timeout_s=60.0)
        _, kwargs = mock_http_cls.call_args
        assert kwargs["config"].timeout_s == 60.0

    def test_default_field_values(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http)
        assert instance.base_url == DEFAULT_BASE_URL
        assert instance.timeout_s == 120.0
        assert instance.model is None
        assert instance.voice is None
        assert instance.response_format is None
        assert instance.speed is None
        assert instance.duration is None
        assert instance.instrumental is None

    def test_api_key_excluded_from_repr(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http)
        assert "api_key" not in repr(instance)

    def test_extra_fields_raise_validation_error(self, mock_http: MagicMock) -> None:
        with pytest.raises(ValidationError):
            _make_tts(mock_http, unknown_param="value")

    def test_speed_field_validator_on_init(self, mock_http: MagicMock) -> None:
        with pytest.raises(ValidationError):
            _make_tts(mock_http, speed=0.0)

    def test_duration_field_validator_on_init(self, mock_http: MagicMock) -> None:
        with pytest.raises(ValidationError):
            _make_tts(mock_http, duration=1.0)


class TestValidateModelId:
    def test_known_model_emits_no_warning(self, mock_http: MagicMock) -> None:
        tts_module._audio_model_ids_cache = ["openai-audio", "tts-1"]
        tts_module._audio_model_ids_loaded = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_tts(mock_http, model="openai-audio")
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_unknown_model_emits_user_warning(self, mock_http: MagicMock) -> None:
        tts_module._audio_model_ids_cache = ["openai-audio", "tts-1"]
        tts_module._audio_model_ids_loaded = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_tts(mock_http, model="ghost-model")
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1

    def test_warning_message_contains_model_name(self, mock_http: MagicMock) -> None:
        tts_module._audio_model_ids_cache = ["openai-audio"]
        tts_module._audio_model_ids_loaded = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_tts(mock_http, model="nonexistent-model")
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert "nonexistent-model" in str(user_warnings[0].message)

    def test_warning_message_contains_catalog_size(self, mock_http: MagicMock) -> None:
        tts_module._audio_model_ids_cache = ["a", "b", "c"]
        tts_module._audio_model_ids_loaded = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_tts(mock_http, model="missing")
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert "3" in str(user_warnings[0].message)

    def test_none_model_emits_no_warning(self, mock_http: MagicMock) -> None:
        tts_module._audio_model_ids_cache = ["openai-audio"]
        tts_module._audio_model_ids_loaded = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_tts(mock_http, model=None)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_empty_cache_emits_no_warning(self, mock_http: MagicMock) -> None:
        tts_module._audio_model_ids_cache = []
        tts_module._audio_model_ids_loaded = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_tts(mock_http, model="any-model")
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_validator_does_not_block_request_for_unknown_model(
        self, mock_http: MagicMock
    ) -> None:
        tts_module._audio_model_ids_cache = ["openai-audio"]
        tts_module._audio_model_ids_loaded = True
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            instance = _make_tts(mock_http, model="ghost-model")
        # La instancia se crea correctamente a pesar del warning.
        assert instance.model == "ghost-model"


class TestDefaultsDict:
    def test_returns_empty_dict_when_no_fields_configured(self, tts: TTSPollinations) -> None:
        assert tts._defaults_dict() == {}

    def test_includes_all_non_none_fields(self, mock_http: MagicMock) -> None:
        instance = _make_tts(
            mock_http,
            model="tts-1",
            voice="rachel",
            response_format="wav",
            speed=1.5,
            duration=30.0,
            instrumental=True,
        )
        result = instance._defaults_dict()
        assert result == {
            "model": "tts-1",
            "voice": "rachel",
            "response_format": "wav",
            "speed": 1.5,
            "duration": 30.0,
            "instrumental": True,
        }

    def test_excludes_none_model(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http, voice="echo")
        assert "model" not in instance._defaults_dict()

    def test_excludes_none_duration(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http, voice="echo")
        assert "duration" not in instance._defaults_dict()

    def test_excludes_none_instrumental(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http, voice="echo")
        assert "instrumental" not in instance._defaults_dict()

    def test_instrumental_false_is_included(self, mock_http: MagicMock) -> None:
        # False no es None; debe aparecer en el dict de defaults.
        instance = _make_tts(mock_http, instrumental=False)
        result = instance._defaults_dict()
        assert "instrumental" in result
        assert result["instrumental"] is False

    def test_partial_config_only_returns_set_fields(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http, voice="shimmer")
        result = instance._defaults_dict()
        assert list(result.keys()) == ["voice"]


class TestBuildBody:
    def test_input_always_set_from_text_argument(self, tts: TTSPollinations) -> None:
        body = tts._build_body("Hello world")
        assert body["input"] == "Hello world"

    def test_instance_defaults_applied_to_body(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http, voice="domi", response_format="wav")
        body = instance._build_body("Test")
        assert body["voice"] == "domi"
        assert body["response_format"] == "wav"

    def test_per_call_params_override_instance_defaults(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http, voice="alloy")
        body = instance._build_body("Test", params={"voice": "echo"})
        assert body["voice"] == "echo"

    def test_per_call_kwargs_override_params(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http, voice="alloy")
        body = instance._build_body("Test", params={"voice": "echo"}, voice="fable")
        assert body["voice"] == "fable"

    def test_text_argument_always_wins_over_input_in_params(
        self, tts: TTSPollinations
    ) -> None:
        body = tts._build_body("Winner", params={"input": "Loser"})
        assert body["input"] == "Winner"

    def test_speech_request_defaults_present_when_no_instance_config(
        self, tts: TTSPollinations
    ) -> None:
        body = tts._build_body("Hello")
        assert body["voice"] == "alloy"
        assert body["response_format"] == "mp3"
        assert body["speed"] == 1.0

    def test_none_fields_excluded_from_body(self, tts: TTSPollinations) -> None:
        body = tts._build_body("Hello")
        assert "model" not in body
        assert "duration" not in body
        assert "instrumental" not in body

    def test_raises_validation_error_on_invalid_speed(self, tts: TTSPollinations) -> None:
        with pytest.raises(ValidationError):
            tts._build_body("Hello", speed=99.0)

    def test_raises_validation_error_on_empty_text(self, tts: TTSPollinations) -> None:
        with pytest.raises(ValidationError):
            tts._build_body("")

    def test_elevenmusic_params_forwarded_correctly(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http, model="elevenmusic", duration=60.0, instrumental=True)
        body = instance._build_body("A peaceful melody")
        assert body["model"] == "elevenmusic"
        assert body["duration"] == 60.0
        assert body["instrumental"] is True

    def test_params_dict_not_mutated(self, tts: TTSPollinations) -> None:
        params = {"voice": "ash"}
        original = dict(params)
        tts._build_body("Hello", params=params)
        assert params == original


class TestGenerateResponse:
    def test_calls_post_json_at_correct_path(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.post_json.return_value = _fake_response()
        tts.generate_response("Hello")
        args, _ = mock_http.post_json.call_args
        assert args[0] == "/v1/audio/speech"

    def test_body_contains_input_field(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.post_json.return_value = _fake_response()
        tts.generate_response("My text")
        args, _ = mock_http.post_json.call_args
        assert args[1]["input"] == "My text"

    def test_returns_httpx_response_object(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        fake_resp = _fake_response()
        mock_http.post_json.return_value = fake_resp
        result = tts.generate_response("Hello")
        assert result is fake_resp

    def test_post_json_called_once(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.post_json.return_value = _fake_response()
        tts.generate_response("Hello")
        mock_http.post_json.assert_called_once()

    def test_per_call_params_reflected_in_body(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.post_json.return_value = _fake_response()
        tts.generate_response("Hello", params={"voice": "sage"})
        args, _ = mock_http.post_json.call_args
        assert args[1]["voice"] == "sage"

    def test_per_call_kwargs_reflected_in_body(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.post_json.return_value = _fake_response()
        tts.generate_response("Hello", response_format="opus")
        args, _ = mock_http.post_json.call_args
        assert args[1]["response_format"] == "opus"


class TestAGenerateResponse:
    async def test_calls_apost_json_at_correct_path(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.apost_json.return_value = _fake_response()
        await tts.agenerate_response("Hello")
        args, _ = mock_http.apost_json.call_args
        assert args[0] == "/v1/audio/speech"

    async def test_body_contains_input_field(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.apost_json.return_value = _fake_response()
        await tts.agenerate_response("Async text")
        args, _ = mock_http.apost_json.call_args
        assert args[1]["input"] == "Async text"

    async def test_returns_httpx_response_object(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        fake_resp = _fake_response()
        mock_http.apost_json.return_value = fake_resp
        result = await tts.agenerate_response("Hello")
        assert result is fake_resp

    async def test_apost_json_called_once(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.apost_json.return_value = _fake_response()
        await tts.agenerate_response("Hello")
        mock_http.apost_json.assert_called_once()

    async def test_per_call_params_reflected_in_body(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.apost_json.return_value = _fake_response()
        await tts.agenerate_response("Hello", params={"voice": "coral"})
        args, _ = mock_http.apost_json.call_args
        assert args[1]["voice"] == "coral"


class TestGenerate:
    def test_returns_bytes_from_response_content(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.post_json.return_value = _fake_response(b"\xff\xfbaudio_bytes")
        result = tts.generate("Hello")
        assert result == b"\xff\xfbaudio_bytes"

    def test_return_type_is_bytes(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.post_json.return_value = _fake_response(b"data")
        assert isinstance(tts.generate("Hello"), bytes)

    def test_delegates_to_generate_response(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        # Verifica la delegación observando que post_json recibe exactamente
        # el path correcto y el body con los kwargs forwarded.
        mock_http.post_json.return_value = _fake_response(b"data")
        tts.generate("Hello", response_format="flac")
        args, _ = mock_http.post_json.call_args
        assert args[0] == "/v1/audio/speech"
        assert args[1]["input"] == "Hello"
        assert args[1]["response_format"] == "flac"

    def test_kwargs_forwarded_to_generate_response(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.post_json.return_value = _fake_response(b"data")
        tts.generate("Hello", voice="ash", speed=2.0)
        args, _ = mock_http.post_json.call_args
        assert args[1]["voice"] == "ash"
        assert args[1]["speed"] == 2.0


class TestAGenerate:
    async def test_returns_bytes_from_response_content(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.apost_json.return_value = _fake_response(b"async_audio")
        result = await tts.agenerate("Hello")
        assert result == b"async_audio"

    async def test_return_type_is_bytes(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.apost_json.return_value = _fake_response(b"x")
        assert isinstance(await tts.agenerate("Hello"), bytes)

    async def test_delegates_to_agenerate_response(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        # Verifica delegación: agenerate llama a apost_json con el body correcto.
        mock_http.apost_json.return_value = _fake_response(b"data")
        await tts.agenerate("Hello", voice="ash")
        args, _ = mock_http.apost_json.call_args
        assert args[0] == "/v1/audio/speech"
        assert args[1]["input"] == "Hello"
        assert args[1]["voice"] == "ash"

    async def test_kwargs_forwarded_to_agenerate_response(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.apost_json.return_value = _fake_response(b"data")
        await tts.agenerate("Hello", response_format="opus")
        args, _ = mock_http.apost_json.call_args
        assert args[1]["response_format"] == "opus"


class TestInvoke:
    def test_returns_bytes(self, tts: TTSPollinations, mock_http: MagicMock) -> None:
        mock_http.post_json.return_value = _fake_response(b"bytes_result")
        result = tts.invoke("Hello")
        assert result == b"bytes_result"

    def test_delegates_to_generate_with_input_text(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        # invoke("Hello from invoke") debe llegar a post_json con input correcto.
        mock_http.post_json.return_value = _fake_response(b"audio")
        tts.invoke("Hello from invoke")
        args, _ = mock_http.post_json.call_args
        assert args[1]["input"] == "Hello from invoke"

    def test_config_parameter_is_ignored(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        # config no debe aparecer en el body enviado a la API.
        mock_http.post_json.return_value = _fake_response(b"audio")
        tts.invoke("Hello", config={"callbacks": [], "run_name": "test"})
        args, _ = mock_http.post_json.call_args
        body = args[1]
        assert "config" not in body

    def test_config_none_is_ignored(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.post_json.return_value = _fake_response(b"audio")
        tts.invoke("Hello", config=None)
        args, _ = mock_http.post_json.call_args
        assert "config" not in args[1]

    def test_kwargs_forwarded_to_generate(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.post_json.return_value = _fake_response(b"audio")
        tts.invoke("Hello", voice="shimmer", response_format="aac")
        args, _ = mock_http.post_json.call_args
        body = args[1]
        assert body["voice"] == "shimmer"
        assert body["response_format"] == "aac"

    def test_config_does_not_leak_into_generate_kwargs(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.post_json.return_value = _fake_response(b"audio")
        tts.invoke("Hello", config={"x": 1}, speed=1.5)
        args, _ = mock_http.post_json.call_args
        body = args[1]
        assert "config" not in body
        assert body["speed"] == 1.5


class TestAInvoke:
    async def test_returns_bytes(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.apost_json.return_value = _fake_response(b"async_bytes")
        result = await tts.ainvoke("Hello")
        assert result == b"async_bytes"

    async def test_delegates_to_agenerate_with_input_text(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        # ainvoke("Async invoke") debe llegar a apost_json con input correcto.
        mock_http.apost_json.return_value = _fake_response(b"audio")
        await tts.ainvoke("Async invoke")
        args, _ = mock_http.apost_json.call_args
        assert args[1]["input"] == "Async invoke"

    async def test_config_parameter_is_ignored(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        # config no debe aparecer en el body enviado a la API.
        mock_http.apost_json.return_value = _fake_response(b"audio")
        await tts.ainvoke("Hello", config={"run_name": "test"})
        args, _ = mock_http.apost_json.call_args
        assert "config" not in args[1]

    async def test_config_none_is_ignored(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.apost_json.return_value = _fake_response(b"audio")
        await tts.ainvoke("Hello", config=None)
        args, _ = mock_http.apost_json.call_args
        assert "config" not in args[1]

    async def test_kwargs_forwarded_to_agenerate(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.apost_json.return_value = _fake_response(b"audio")
        await tts.ainvoke("Hello", speed=2.0, voice="ballad")
        args, _ = mock_http.apost_json.call_args
        body = args[1]
        assert body["speed"] == 2.0
        assert body["voice"] == "ballad"

    async def test_config_does_not_leak_into_agenerate_kwargs(
        self, tts: TTSPollinations, mock_http: MagicMock
    ) -> None:
        mock_http.apost_json.return_value = _fake_response(b"audio")
        await tts.ainvoke("Hello", config={"x": 1}, response_format="wav")
        args, _ = mock_http.apost_json.call_args
        body = args[1]
        assert "config" not in body
        assert body["response_format"] == "wav"


class TestWithParams:
    def test_returns_new_instance_not_same_object(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http)
        with _tts_patches(mock_http):
            new_instance = instance.with_params(voice="echo")
        assert new_instance is not instance

    def test_original_instance_unchanged_after_with_params(
        self, mock_http: MagicMock
    ) -> None:
        instance = _make_tts(mock_http, voice="alloy")
        with _tts_patches(mock_http):
            instance.with_params(voice="echo")
        assert instance.voice == "alloy"

    def test_override_field_applied_in_new_instance(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http, voice="alloy")
        with _tts_patches(mock_http):
            new_instance = instance.with_params(voice="echo")
        assert new_instance.voice == "echo"

    def test_non_overridden_fields_preserved(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http, response_format="wav", speed=2.0)
        with _tts_patches(mock_http):
            new_instance = instance.with_params(voice="coral")
        assert new_instance.response_format == "wav"
        assert new_instance.speed == 2.0

    def test_api_key_preserved_in_new_instance(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http)
        with _tts_patches(mock_http):
            new_instance = instance.with_params(voice="shimmer")
        assert new_instance.api_key == instance.api_key

    def test_base_url_preserved_in_new_instance(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http)
        with _tts_patches(mock_http):
            new_instance = instance.with_params(voice="echo")
        assert new_instance.base_url == instance.base_url

    def test_new_instance_is_tts_pollinations_type(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http)
        with _tts_patches(mock_http):
            new_instance = instance.with_params()
        assert isinstance(new_instance, TTSPollinations)

    def test_multiple_overrides_applied_simultaneously(
        self, mock_http: MagicMock
    ) -> None:
        instance = _make_tts(mock_http)
        with _tts_patches(mock_http):
            new_instance = instance.with_params(
                voice="sage", response_format="flac", speed=1.25
            )
        assert new_instance.voice == "sage"
        assert new_instance.response_format == "flac"
        assert new_instance.speed == 1.25

    def test_can_set_field_to_none_via_override(self, mock_http: MagicMock) -> None:
        instance = _make_tts(mock_http, voice="alloy")
        with _tts_patches(mock_http):
            new_instance = instance.with_params(voice=None)
        assert new_instance.voice is None
