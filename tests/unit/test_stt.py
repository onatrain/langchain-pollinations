from __future__ import annotations

import json
import warnings as _warnings_module
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from pydantic import ValidationError

from langchain_pollinations.stt import (
    _AUDIO_MIME_TYPES,
    _FALLBACK_MIME_TYPE,
    AudioInputFormat,
    STTPollinations,
    TranscriptionFormat,
    TranscriptionParams,
    TranscriptionResponse,
)

_DUMMY_AUDIO = b"\xff\xfb\x90\x00" * 64  # bytes con cabecera pseudo-MP3
_DUMMY_API_KEY = "sk-test-abc123"


def _make_httpx_json_response(body: dict, status_code: int = 200) -> httpx.Response:
    """Construye un httpx.Response con body JSON y status dado."""
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )


def _make_httpx_text_response(text: str, status_code: int = 200) -> httpx.Response:
    """Construye un httpx.Response con body texto plano y status dado."""
    return httpx.Response(
        status_code=status_code,
        content=text.encode(),
        headers={"content-type": "text/plain"},
    )


def _make_stt(**kwargs) -> tuple[STTPollinations, MagicMock]:
    """
    Instancia STTPollinations con todas las dependencias externas mockeadas.

    Retorna (client, mock_http_instance). El mock_http_instance tiene
    post_multipart como MagicMock y apost_multipart como AsyncMock.
    """
    mock_http = MagicMock()
    mock_http.post_multipart = MagicMock()
    mock_http.apost_multipart = AsyncMock()

    with (
        patch("langchain_pollinations.stt._load_audio_model_ids"),
        patch("langchain_pollinations.stt.AuthConfig") as mock_auth_cls,
        patch("langchain_pollinations.stt.PollinationsHttpClient", return_value=mock_http),
    ):
        mock_auth_cls.from_env_or_value.return_value = MagicMock(api_key=_DUMMY_API_KEY)
        client = STTPollinations(**kwargs)

    return client, mock_http


@pytest.fixture()
def stt_client() -> tuple[STTPollinations, MagicMock]:
    """Instancia STTPollinations con configuración mínima (todos los campos en None)."""
    return _make_stt()


@pytest.fixture()
def stt_client_persistent():
    """
    Instancia STTPollinations manteniendo los patches activos durante todo
    el cuerpo del test. Usar en lugar de stt_client cuando el test llama a
    with_params() o cualquier otro método que cree una nueva instancia
    de STTPollinations internamente.
    """
    mock_http = MagicMock()
    mock_http.post_multipart = MagicMock()
    mock_http.apost_multipart = AsyncMock()

    with (
        patch("langchain_pollinations.stt._load_audio_model_ids"),
        patch("langchain_pollinations.stt.AuthConfig") as mock_auth_cls,
        patch("langchain_pollinations.stt.PollinationsHttpClient", return_value=mock_http),
    ):
        mock_auth_cls.from_env_or_value.return_value = MagicMock(api_key=_DUMMY_API_KEY)
        client = STTPollinations()
        yield client, mock_http


@pytest.fixture()
def json_resp() -> httpx.Response:
    return _make_httpx_json_response({"text": "Hello world"})


@pytest.fixture()
def verbose_json_resp() -> httpx.Response:
    return _make_httpx_json_response(
        {
            "text": "Hello world",
            "language": "en",
            "duration": 3.5,
            "segments": [{"id": 0, "text": "Hello world", "start": 0.0, "end": 3.5}],
        }
    )


class TestAudioMimeTypesConstant:
    def test_mp3_maps_to_audio_mpeg(self):
        assert _AUDIO_MIME_TYPES["mp3"] == "audio/mpeg"

    def test_mp4_maps_to_audio_mp4(self):
        assert _AUDIO_MIME_TYPES["mp4"] == "audio/mp4"

    def test_mpeg_maps_to_audio_mpeg(self):
        assert _AUDIO_MIME_TYPES["mpeg"] == "audio/mpeg"

    def test_mpga_maps_to_audio_mpeg(self):
        assert _AUDIO_MIME_TYPES["mpga"] == "audio/mpeg"

    def test_m4a_maps_to_audio_mp4(self):
        assert _AUDIO_MIME_TYPES["m4a"] == "audio/mp4"

    def test_wav_maps_to_audio_wav(self):
        assert _AUDIO_MIME_TYPES["wav"] == "audio/wav"

    def test_webm_maps_to_audio_webm(self):
        assert _AUDIO_MIME_TYPES["webm"] == "audio/webm"

    def test_all_audio_input_format_extensions_present(self):
        # AudioInputFormat = Literal["mp3","mp4","mpeg","mpga","m4a","wav","webm"]
        expected = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}
        assert expected.issubset(set(_AUDIO_MIME_TYPES.keys()))

    def test_fallback_mime_type_value(self):
        assert _FALLBACK_MIME_TYPE == "application/octet-stream"


class TestTranscriptionParams:
    def test_default_model(self):
        assert TranscriptionParams().model == "whisper-large-v3"

    def test_default_response_format(self):
        assert TranscriptionParams().response_format == "json"

    def test_default_language_is_none(self):
        assert TranscriptionParams().language is None

    def test_default_prompt_is_none(self):
        assert TranscriptionParams().prompt is None

    def test_default_temperature_is_none(self):
        assert TranscriptionParams().temperature is None

    # to_form_data — cobertura de inclusión/exclusión

    def test_to_form_data_default_contains_only_model_and_format(self):
        fd = TranscriptionParams().to_form_data()
        assert set(fd.keys()) == {"model", "response_format"}

    def test_to_form_data_model_value(self):
        fd = TranscriptionParams().to_form_data()
        assert fd["model"] == "whisper-large-v3"

    def test_to_form_data_response_format_value(self):
        fd = TranscriptionParams().to_form_data()
        assert fd["response_format"] == "json"

    def test_to_form_data_excludes_none_language(self):
        fd = TranscriptionParams(language=None).to_form_data()
        assert "language" not in fd

    def test_to_form_data_excludes_none_prompt(self):
        fd = TranscriptionParams(prompt=None).to_form_data()
        assert "prompt" not in fd

    def test_to_form_data_excludes_none_temperature(self):
        fd = TranscriptionParams(temperature=None).to_form_data()
        assert "temperature" not in fd

    def test_to_form_data_includes_language_when_set(self):
        fd = TranscriptionParams(language="es").to_form_data()
        assert fd["language"] == "es"

    def test_to_form_data_includes_prompt_when_set(self):
        fd = TranscriptionParams(prompt="continue this").to_form_data()
        assert fd["prompt"] == "continue this"

    def test_to_form_data_includes_temperature_when_set(self):
        fd = TranscriptionParams(temperature=0.5).to_form_data()
        assert "temperature" in fd

    def test_to_form_data_all_values_are_str(self):
        fd = TranscriptionParams(language="en", temperature=0.4).to_form_data()
        for k, v in fd.items():
            assert isinstance(v, str), f"Campo '{k}' tiene valor no-str: {type(v)}"

    def test_to_form_data_temperature_float_serialized_as_str(self):
        fd = TranscriptionParams(temperature=0.3).to_form_data()
        assert isinstance(fd["temperature"], str)
        assert fd["temperature"] == "0.3"

    def test_to_form_data_full_params_all_present(self):
        fd = TranscriptionParams(
            model="scribe",
            language="fr",
            prompt="contexto",
            response_format="srt",
            temperature=0.2,
        ).to_form_data()
        assert fd["model"] == "scribe"
        assert fd["language"] == "fr"
        assert fd["prompt"] == "contexto"
        assert fd["response_format"] == "srt"
        assert fd["temperature"] == "0.2"

    # response_format — formatos válidos

    def test_response_format_json(self):
        assert TranscriptionParams(response_format="json").response_format == "json"

    def test_response_format_text(self):
        assert TranscriptionParams(response_format="text").response_format == "text"

    def test_response_format_srt(self):
        assert TranscriptionParams(response_format="srt").response_format == "srt"

    def test_response_format_verbose_json(self):
        assert TranscriptionParams(response_format="verbose_json").response_format == "verbose_json"

    def test_response_format_vtt(self):
        assert TranscriptionParams(response_format="vtt").response_format == "vtt"

    def test_invalid_response_format_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TranscriptionParams(response_format="xml")

    # temperature — límites y validación

    def test_temperature_at_zero(self):
        assert TranscriptionParams(temperature=0.0).temperature == 0.0

    def test_temperature_at_one(self):
        assert TranscriptionParams(temperature=1.0).temperature == 1.0

    def test_temperature_below_zero_raises(self):
        with pytest.raises(ValidationError):
            TranscriptionParams(temperature=-0.01)

    def test_temperature_above_one_raises(self):
        with pytest.raises(ValidationError):
            TranscriptionParams(temperature=1.01)

    # extra="forbid"

    def test_unknown_field_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TranscriptionParams(unknown_param="value")

    # modelos conocidos del API

    def test_whisper_large_v3_accepted(self):
        assert TranscriptionParams(model="whisper-large-v3").model == "whisper-large-v3"

    def test_whisper_1_accepted(self):
        assert TranscriptionParams(model="whisper-1").model == "whisper-1"

    def test_scribe_accepted(self):
        assert TranscriptionParams(model="scribe").model == "scribe"


class TestTranscriptionResponse:
    def test_construction_with_text(self):
        r = TranscriptionResponse(text="Hello world")
        assert r.text == "Hello world"

    def test_text_field_required(self):
        with pytest.raises(ValidationError):
            TranscriptionResponse()

    def test_empty_string_accepted(self):
        assert TranscriptionResponse(text="").text == ""

    def test_extra_fields_not_rejected(self):
        r = TranscriptionResponse(text="Hi", language="fr", duration=2.1)
        assert r.text == "Hi"

    def test_extra_fields_in_model_extra(self):
        r = TranscriptionResponse(text="Hi", language="fr")
        assert r.model_extra["language"] == "fr"

    def test_extra_float_field_in_model_extra(self):
        r = TranscriptionResponse(text="Hi", duration=3.14)
        assert r.model_extra["duration"] == 3.14

    def test_extra_list_field_in_model_extra(self):
        r = TranscriptionResponse(text="Hi", segments=[{"id": 0}])
        assert isinstance(r.model_extra["segments"], list)

    def test_model_validate_basic_dict(self):
        r = TranscriptionResponse.model_validate({"text": "Test"})
        assert r.text == "Test"

    def test_model_validate_verbose_json_payload(self):
        data = {
            "text": "Hello world",
            "language": "en",
            "duration": 3.14,
            "segments": [{"id": 0, "text": "Hello world"}],
            "words": [],
        }
        r = TranscriptionResponse.model_validate(data)
        assert r.text == "Hello world"
        assert r.model_extra["language"] == "en"
        assert r.model_extra["duration"] == 3.14
        assert isinstance(r.model_extra["segments"], list)

    def test_model_validate_missing_text_raises(self):
        with pytest.raises(ValidationError):
            TranscriptionResponse.model_validate({"language": "en"})


class TestMimeTypeFor:
    def test_mp3_returns_audio_mpeg(self):
        assert STTPollinations._mime_type_for("audio.mp3") == "audio/mpeg"

    def test_mp4_returns_audio_mp4(self):
        assert STTPollinations._mime_type_for("audio.mp4") == "audio/mp4"

    def test_mpeg_returns_audio_mpeg(self):
        assert STTPollinations._mime_type_for("audio.mpeg") == "audio/mpeg"

    def test_mpga_returns_audio_mpeg(self):
        assert STTPollinations._mime_type_for("audio.mpga") == "audio/mpeg"

    def test_m4a_returns_audio_mp4(self):
        assert STTPollinations._mime_type_for("audio.m4a") == "audio/mp4"

    def test_wav_returns_audio_wav(self):
        assert STTPollinations._mime_type_for("recording.wav") == "audio/wav"

    def test_webm_returns_audio_webm(self):
        assert STTPollinations._mime_type_for("clip.webm") == "audio/webm"

    def test_unknown_extension_returns_fallback(self):
        assert STTPollinations._mime_type_for("audio.ogg") == _FALLBACK_MIME_TYPE

    def test_no_extension_returns_fallback(self):
        assert STTPollinations._mime_type_for("audiofile") == _FALLBACK_MIME_TYPE

    def test_uppercase_extension_normalized(self):
        assert STTPollinations._mime_type_for("audio.MP3") == "audio/mpeg"

    def test_mixed_case_extension_normalized(self):
        assert STTPollinations._mime_type_for("audio.Wav") == "audio/wav"

    def test_filename_starting_with_dot(self):
        # os.path.splitext(".mp3") == ('.mp3', '') → sin extensión → fallback
        assert STTPollinations._mime_type_for(".mp3") == _FALLBACK_MIME_TYPE

    def test_absolute_path_with_wav(self):
        assert STTPollinations._mime_type_for("/tmp/recordings/speech.wav") == "audio/wav"

    def test_filename_with_multiple_dots(self):
        assert STTPollinations._mime_type_for("my.audio.recording.mp3") == "audio/mpeg"

    def test_empty_string_returns_fallback(self):
        assert STTPollinations._mime_type_for("") == _FALLBACK_MIME_TYPE


class TestSTTPollinationsInit:
    def test_calls_load_audio_model_ids_once(self):
        mock_http = MagicMock()
        with (
            patch("langchain_pollinations.stt._load_audio_model_ids") as mock_load,
            patch("langchain_pollinations.stt.AuthConfig") as mock_auth_cls,
            patch("langchain_pollinations.stt.PollinationsHttpClient", return_value=mock_http),
        ):
            mock_auth_cls.from_env_or_value.return_value = MagicMock(api_key="k")
            STTPollinations()
        mock_load.assert_called_once()

    def test_passes_none_api_key_to_load_when_not_provided(self):
        mock_http = MagicMock()
        with (
            patch("langchain_pollinations.stt._load_audio_model_ids") as mock_load,
            patch("langchain_pollinations.stt.AuthConfig") as mock_auth_cls,
            patch("langchain_pollinations.stt.PollinationsHttpClient", return_value=mock_http),
        ):
            mock_auth_cls.from_env_or_value.return_value = MagicMock(api_key="k")
            STTPollinations()
        mock_load.assert_called_once_with(None)

    def test_passes_api_key_to_load_when_provided(self):
        mock_http = MagicMock()
        with (
            patch("langchain_pollinations.stt._load_audio_model_ids") as mock_load,
            patch("langchain_pollinations.stt.AuthConfig") as mock_auth_cls,
            patch("langchain_pollinations.stt.PollinationsHttpClient", return_value=mock_http),
        ):
            mock_auth_cls.from_env_or_value.return_value = MagicMock(api_key="my-key")
            STTPollinations(api_key="my-key")
        mock_load.assert_called_once_with("my-key")

    def test_http_client_is_assigned(self, stt_client):
        client, mock_http = stt_client
        assert client._http is mock_http

    def test_default_base_url(self, stt_client):
        client, _ = stt_client
        assert client.base_url == "https://gen.pollinations.ai"

    def test_default_timeout_s(self, stt_client):
        client, _ = stt_client
        assert client.timeout_s == 120.0

    def test_default_file_name(self, stt_client):
        client, _ = stt_client
        assert client.file_name == "audio.mp3"

    def test_default_model_is_none(self, stt_client):
        client, _ = stt_client
        assert client.model is None

    def test_default_language_is_none(self, stt_client):
        client, _ = stt_client
        assert client.language is None

    def test_default_prompt_is_none(self, stt_client):
        client, _ = stt_client
        assert client.prompt is None

    def test_default_response_format_is_none(self, stt_client):
        client, _ = stt_client
        assert client.response_format is None

    def test_default_temperature_is_none(self, stt_client):
        client, _ = stt_client
        assert client.temperature is None

    def test_custom_model_set(self):
        client, _ = _make_stt(model="whisper-1")
        assert client.model == "whisper-1"

    def test_custom_language_set(self):
        client, _ = _make_stt(language="en")
        assert client.language == "en"

    def test_custom_response_format_set(self):
        client, _ = _make_stt(response_format="text")
        assert client.response_format == "text"

    def test_custom_file_name_set(self):
        client, _ = _make_stt(file_name="speech.wav")
        assert client.file_name == "speech.wav"

    def test_custom_temperature_set(self):
        client, _ = _make_stt(temperature=0.4)
        assert client.temperature == 0.4

    def test_unknown_field_raises(self):
        with pytest.raises(Exception):
            _make_stt(nonexistent_field="value")


class TestValidateModelId:
    def _make_with_patched_cache(self, cache_contents: list, model: str | None):
        """Helper que parchea el caché y crea el cliente, capturando warnings."""
        with (
            patch("langchain_pollinations.stt._audio_model_ids_cache", cache_contents),
            patch("langchain_pollinations.stt._load_audio_model_ids"),
            patch("langchain_pollinations.stt.AuthConfig") as mock_auth,
            patch("langchain_pollinations.stt.PollinationsHttpClient"),
        ):
            mock_auth.from_env_or_value.return_value = MagicMock(api_key="k")
            with _warnings_module.catch_warnings(record=True) as recorded:
                _warnings_module.simplefilter("always")
                client = STTPollinations(model=model)
        return client, recorded

    def test_known_model_emits_no_catalog_warning(self):
        _, recorded = self._make_with_patched_cache(
            ["whisper-large-v3", "whisper-1"], "whisper-large-v3"
        )
        catalog_warnings = [
            w for w in recorded
            if issubclass(w.category, UserWarning) and "catalog" in str(w.message)
        ]
        assert len(catalog_warnings) == 0

    def test_unknown_model_emits_user_warning(self):
        with (
            patch("langchain_pollinations.stt._audio_model_ids_cache", ["whisper-large-v3"]),
            patch("langchain_pollinations.stt._load_audio_model_ids"),
            patch("langchain_pollinations.stt.AuthConfig") as mock_auth,
            patch("langchain_pollinations.stt.PollinationsHttpClient"),
            pytest.warns(UserWarning, match="not in the known audio model catalog"),
        ):
            mock_auth.from_env_or_value.return_value = MagicMock(api_key="k")
            STTPollinations(model="gpt-5o-audio")

    def test_warning_message_contains_model_name(self):
        with (
            patch("langchain_pollinations.stt._audio_model_ids_cache", ["whisper-large-v3"]),
            patch("langchain_pollinations.stt._load_audio_model_ids"),
            patch("langchain_pollinations.stt.AuthConfig") as mock_auth,
            patch("langchain_pollinations.stt.PollinationsHttpClient"),
            pytest.warns(UserWarning) as warning_list,
        ):
            mock_auth.from_env_or_value.return_value = MagicMock(api_key="k")
            STTPollinations(model="my-custom-model")
        assert "my-custom-model" in str(warning_list[0].message)

    def test_empty_cache_emits_no_warning_for_any_model(self):
        _, recorded = self._make_with_patched_cache([], "gpt-turbo-stt")
        catalog_warnings = [
            w for w in recorded
            if issubclass(w.category, UserWarning) and "catalog" in str(w.message)
        ]
        assert len(catalog_warnings) == 0

    def test_unknown_model_does_not_block_instantiation(self):
        with (
            patch("langchain_pollinations.stt._audio_model_ids_cache", ["whisper-large-v3"]),
            patch("langchain_pollinations.stt._load_audio_model_ids"),
            patch("langchain_pollinations.stt.AuthConfig") as mock_auth,
            patch("langchain_pollinations.stt.PollinationsHttpClient"),
            pytest.warns(UserWarning),
        ):
            mock_auth.from_env_or_value.return_value = MagicMock(api_key="k")
            client = STTPollinations(model="unknown")
        assert client.model == "unknown"

    def test_warning_mentions_catalog_size(self):
        cache = ["whisper-large-v3", "whisper-1", "scribe"]
        with (
            patch("langchain_pollinations.stt._audio_model_ids_cache", cache),
            patch("langchain_pollinations.stt._load_audio_model_ids"),
            patch("langchain_pollinations.stt.AuthConfig") as mock_auth,
            patch("langchain_pollinations.stt.PollinationsHttpClient"),
            pytest.warns(UserWarning) as warning_list,
        ):
            mock_auth.from_env_or_value.return_value = MagicMock(api_key="k")
            STTPollinations(model="nonexistent")
        assert "3" in str(warning_list[0].message)


class TestDefaultsDict:
    def test_all_none_returns_empty_dict(self, stt_client):
        client, _ = stt_client
        assert client._defaults_dict() == {}

    def test_model_included_when_set(self):
        client, _ = _make_stt(model="whisper-1")
        assert client._defaults_dict()["model"] == "whisper-1"

    def test_language_included_when_set(self):
        client, _ = _make_stt(language="es")
        assert client._defaults_dict()["language"] == "es"

    def test_prompt_included_when_set(self):
        client, _ = _make_stt(prompt="hint")
        assert client._defaults_dict()["prompt"] == "hint"

    def test_response_format_included_when_set(self):
        client, _ = _make_stt(response_format="text")
        assert client._defaults_dict()["response_format"] == "text"

    def test_temperature_included_when_set(self):
        client, _ = _make_stt(temperature=0.3)
        assert client._defaults_dict()["temperature"] == 0.3

    def test_file_name_never_included(self):
        client, _ = _make_stt(file_name="recording.wav")
        assert "file_name" not in client._defaults_dict()

    def test_none_value_excluded(self):
        client, _ = _make_stt(model="whisper-1", language=None)
        d = client._defaults_dict()
        assert "model" in d
        assert "language" not in d

    def test_all_fields_set_returns_full_dict(self):
        client, _ = _make_stt(
            model="scribe",
            language="en",
            prompt="context",
            response_format="srt",
            temperature=0.0,
        )
        assert client._defaults_dict() == {
            "model": "scribe",
            "language": "en",
            "prompt": "context",
            "response_format": "srt",
            "temperature": 0.0,
        }


class TestWithParams:
    def test_returns_new_instance(self, stt_client_persistent):
        client, _ = stt_client_persistent
        assert client.with_params(language="fr") is not client

    def test_override_is_applied(self, stt_client_persistent):
        client, _ = stt_client_persistent
        assert client.with_params(language="fr").language == "fr"

    def test_original_instance_not_modified(self, stt_client_persistent):
        client, _ = stt_client_persistent
        client.with_params(language="fr")
        assert client.language is None

    def test_api_key_preserved(self, stt_client_persistent):
        # api_key se excluye del dump; with_params lo debe reinyectar.
        # stt_client_persistent tiene api_key=None (no se pasa al constructor),
        # pero AuthConfig está mockeado así que with_params crea la instancia sin error.
        client, _ = stt_client_persistent
        new_client = client.with_params(language="es")
        # api_key es None porque _make_stt no lo pasa, pero la instancia se crea correctamente.
        assert new_client.api_key == client.api_key

    def test_multiple_overrides_applied(self, stt_client_persistent):
        client, _ = stt_client_persistent
        new_client = client.with_params(model="scribe", language="de", temperature=0.2)
        assert new_client.model == "scribe"
        assert new_client.language == "de"
        assert new_client.temperature == 0.2

    def test_non_overridden_fields_preserved(self, stt_client_persistent):
        client, _ = stt_client_persistent
        # Configurar campos en la instancia base vía with_params en cadena.
        base = client.with_params(model="whisper-1", language="en")
        result = base.with_params(temperature=0.5)
        assert result.model == "whisper-1"
        assert result.language == "en"
        assert result.temperature == 0.5

    def test_file_name_overridable(self, stt_client_persistent):
        client, _ = stt_client_persistent
        assert client.with_params(file_name="speech.wav").file_name == "speech.wav"

    def test_returns_stt_pollinations_instance(self, stt_client_persistent):
        client, _ = stt_client_persistent
        assert isinstance(client.with_params(language="ja"), STTPollinations)


class TestBuildMultipart:
    def test_returns_three_tuple(self, stt_client):
        client, _ = stt_client
        result = client._build_multipart(_DUMMY_AUDIO)
        assert len(result) == 3

    def test_files_dict_has_file_key(self, stt_client):
        client, _ = stt_client
        files_dict, _, _ = client._build_multipart(_DUMMY_AUDIO)
        assert "file" in files_dict

    def test_files_dict_file_value_is_three_tuple(self, stt_client):
        client, _ = stt_client
        files_dict, _, _ = client._build_multipart(_DUMMY_AUDIO)
        assert len(files_dict["file"]) == 3

    def test_audio_bytes_preserved_in_files_dict(self, stt_client):
        client, _ = stt_client
        files_dict, _, _ = client._build_multipart(_DUMMY_AUDIO)
        _, content, _ = files_dict["file"]
        assert content is _DUMMY_AUDIO

    def test_default_file_name_from_instance(self, stt_client):
        client, _ = stt_client  # default file_name = "audio.mp3"
        files_dict, _, _ = client._build_multipart(_DUMMY_AUDIO)
        filename, _, _ = files_dict["file"]
        assert filename == "audio.mp3"

    def test_default_mime_type_for_mp3(self, stt_client):
        client, _ = stt_client
        files_dict, _, _ = client._build_multipart(_DUMMY_AUDIO)
        _, _, mime = files_dict["file"]
        assert mime == "audio/mpeg"

    def test_file_name_from_kwargs_overrides_instance(self, stt_client):
        client, _ = stt_client
        files_dict, _, _ = client._build_multipart(_DUMMY_AUDIO, file_name="speech.wav")
        filename, _, mime = files_dict["file"]
        assert filename == "speech.wav"
        assert mime == "audio/wav"

    def test_file_name_from_params_dict_overrides_instance(self, stt_client):
        client, _ = stt_client
        files_dict, _, _ = client._build_multipart(
            _DUMMY_AUDIO, params={"file_name": "clip.webm"}
        )
        filename, _, mime = files_dict["file"]
        assert filename == "clip.webm"
        assert mime == "audio/webm"

    def test_kwargs_file_name_overrides_params_file_name(self, stt_client):
        client, _ = stt_client
        files_dict, _, _ = client._build_multipart(
            _DUMMY_AUDIO,
            params={"file_name": "params.mp3"},
            file_name="kwargs.wav",
        )
        filename, _, _ = files_dict["file"]
        assert filename == "kwargs.wav"

    def test_file_name_not_in_form_data(self, stt_client):
        client, _ = stt_client
        _, form_data, _ = client._build_multipart(_DUMMY_AUDIO, file_name="audio.wav")
        assert "file_name" not in form_data

    def test_form_data_contains_model_key(self, stt_client):
        client, _ = stt_client
        _, form_data, _ = client._build_multipart(_DUMMY_AUDIO)
        assert "model" in form_data

    def test_form_data_all_values_are_strings(self, stt_client):
        client, _ = stt_client
        _, form_data, _ = client._build_multipart(_DUMMY_AUDIO)
        for k, v in form_data.items():
            assert isinstance(v, str), f"form_data['{k}'] no es str: {type(v)}"

    def test_third_element_is_transcription_params(self, stt_client):
        client, _ = stt_client
        _, _, params_obj = client._build_multipart(_DUMMY_AUDIO)
        assert isinstance(params_obj, TranscriptionParams)

    def test_params_obj_response_format_defaults_to_json(self, stt_client):
        client, _ = stt_client
        _, _, params_obj = client._build_multipart(_DUMMY_AUDIO)
        assert params_obj.response_format == "json"

    def test_instance_defaults_applied_to_form_data(self):
        client, _ = _make_stt(model="scribe", language="fr", response_format="text")
        _, form_data, params_obj = client._build_multipart(_DUMMY_AUDIO)
        assert form_data["model"] == "scribe"
        assert form_data["language"] == "fr"
        assert params_obj.response_format == "text"

    def test_per_call_params_override_instance_defaults(self):
        client, _ = _make_stt(model="whisper-1", language="en")
        _, form_data, _ = client._build_multipart(
            _DUMMY_AUDIO, params={"model": "scribe", "language": "fr"}
        )
        assert form_data["model"] == "scribe"
        assert form_data["language"] == "fr"

    def test_per_call_kwargs_override_params_dict(self):
        client, _ = _make_stt()
        _, form_data, _ = client._build_multipart(
            _DUMMY_AUDIO,
            params={"model": "whisper-1"},
            model="scribe",
        )
        assert form_data["model"] == "scribe"

    def test_per_call_kwargs_override_instance_defaults(self):
        client, _ = _make_stt(model="whisper-1")
        _, form_data, _ = client._build_multipart(_DUMMY_AUDIO, model="scribe")
        assert form_data["model"] == "scribe"

    def test_none_params_dict_accepted(self, stt_client):
        client, _ = stt_client
        _, _, params_obj = client._build_multipart(_DUMMY_AUDIO, None)
        assert params_obj.model == "whisper-large-v3"

    def test_invalid_temperature_raises_validation_error(self, stt_client):
        client, _ = stt_client
        with pytest.raises(ValidationError):
            client._build_multipart(_DUMMY_AUDIO, temperature=999.0)

    def test_unknown_form_field_raises_validation_error(self, stt_client):
        client, _ = stt_client
        with pytest.raises(ValidationError):
            client._build_multipart(_DUMMY_AUDIO, nonexistent_field="value")

    def test_instance_file_name_used_when_not_overridden(self):
        client, _ = _make_stt(file_name="default.m4a")
        files_dict, _, _ = client._build_multipart(_DUMMY_AUDIO)
        filename, _, mime = files_dict["file"]
        assert filename == "default.m4a"
        assert mime == "audio/mp4"

    def test_params_obj_model_matches_form_data_model(self, stt_client):
        client, _ = stt_client
        _, form_data, params_obj = client._build_multipart(_DUMMY_AUDIO, model="scribe")
        assert params_obj.model == form_data["model"]


class TestParseTranscriptionResponse:
    def test_json_format_returns_transcription_response(self):
        resp = _make_httpx_json_response({"text": "Hello"})
        result = STTPollinations._parse_transcription_response(resp, "json")
        assert isinstance(result, TranscriptionResponse)

    def test_json_format_text_field_correct(self):
        resp = _make_httpx_json_response({"text": "Exact text"})
        result = STTPollinations._parse_transcription_response(resp, "json")
        assert result.text == "Exact text"

    def test_verbose_json_format_returns_transcription_response(self):
        resp = _make_httpx_json_response({"text": "Hello", "language": "en", "duration": 2.0})
        result = STTPollinations._parse_transcription_response(resp, "verbose_json")
        assert isinstance(result, TranscriptionResponse)

    def test_verbose_json_extra_fields_in_model_extra(self):
        resp = _make_httpx_json_response({"text": "Hi", "language": "en", "duration": 2.0})
        result = STTPollinations._parse_transcription_response(resp, "verbose_json")
        assert result.model_extra["language"] == "en"
        assert result.model_extra["duration"] == 2.0

    def test_text_format_returns_str(self):
        resp = _make_httpx_text_response("Hello world")
        result = STTPollinations._parse_transcription_response(resp, "text")
        assert isinstance(result, str)

    def test_text_format_content_correct(self):
        resp = _make_httpx_text_response("Hello world")
        result = STTPollinations._parse_transcription_response(resp, "text")
        assert result == "Hello world"

    def test_srt_format_returns_str(self):
        srt = "1\n00:00:00,000 --> 00:00:02,000\nHello\n"
        resp = _make_httpx_text_response(srt)
        result = STTPollinations._parse_transcription_response(resp, "srt")
        assert isinstance(result, str)
        assert result == srt

    def test_vtt_format_returns_str(self):
        vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello\n"
        resp = _make_httpx_text_response(vtt)
        result = STTPollinations._parse_transcription_response(resp, "vtt")
        assert isinstance(result, str)
        assert result == vtt


class TestTranscribeResponse:
    def test_calls_post_multipart_once(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.transcribe_response(_DUMMY_AUDIO)
        mock_http.post_multipart.assert_called_once()

    def test_calls_correct_endpoint_path(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.transcribe_response(_DUMMY_AUDIO)
        path = mock_http.post_multipart.call_args.args[0]
        assert path == "/v1/audio/transcriptions"

    def test_passes_files_kwarg(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.transcribe_response(_DUMMY_AUDIO)
        assert "files" in mock_http.post_multipart.call_args.kwargs

    def test_passes_data_kwarg(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.transcribe_response(_DUMMY_AUDIO)
        assert "data" in mock_http.post_multipart.call_args.kwargs

    def test_files_kwarg_has_file_key(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.transcribe_response(_DUMMY_AUDIO)
        files = mock_http.post_multipart.call_args.kwargs["files"]
        assert "file" in files

    def test_returns_httpx_response(self, stt_client):
        client, mock_http = stt_client
        expected = _make_httpx_json_response({"text": "hi"})
        mock_http.post_multipart.return_value = expected
        result = client.transcribe_response(_DUMMY_AUDIO)
        assert result is expected

    def test_per_call_params_forwarded_to_data(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.transcribe_response(_DUMMY_AUDIO, params={"language": "de"})
        data = mock_http.post_multipart.call_args.kwargs["data"]
        assert data.get("language") == "de"

    def test_per_call_kwargs_forwarded_to_data(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.transcribe_response(_DUMMY_AUDIO, language="it")
        data = mock_http.post_multipart.call_args.kwargs["data"]
        assert data.get("language") == "it"

    def test_audio_bytes_forwarded_correctly(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.transcribe_response(_DUMMY_AUDIO)
        files = mock_http.post_multipart.call_args.kwargs["files"]
        _, content, _ = files["file"]
        assert content is _DUMMY_AUDIO


@pytest.mark.asyncio
class TestATranscribeResponse:
    async def test_calls_apost_multipart_once(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.atranscribe_response(_DUMMY_AUDIO)
        mock_http.apost_multipart.assert_called_once()

    async def test_calls_correct_endpoint_path(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.atranscribe_response(_DUMMY_AUDIO)
        path = mock_http.apost_multipart.call_args.args[0]
        assert path == "/v1/audio/transcriptions"

    async def test_passes_files_kwarg(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.atranscribe_response(_DUMMY_AUDIO)
        assert "files" in mock_http.apost_multipart.call_args.kwargs

    async def test_passes_data_kwarg(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.atranscribe_response(_DUMMY_AUDIO)
        assert "data" in mock_http.apost_multipart.call_args.kwargs

    async def test_returns_httpx_response(self, stt_client):
        client, mock_http = stt_client
        expected = _make_httpx_json_response({"text": "hi"})
        mock_http.apost_multipart.return_value = expected
        result = await client.atranscribe_response(_DUMMY_AUDIO)
        assert result is expected

    async def test_audio_bytes_forwarded_correctly(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.atranscribe_response(_DUMMY_AUDIO)
        files = mock_http.apost_multipart.call_args.kwargs["files"]
        _, content, _ = files["file"]
        assert content is _DUMMY_AUDIO

    async def test_per_call_params_forwarded(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.atranscribe_response(_DUMMY_AUDIO, params={"language": "pt"})
        data = mock_http.apost_multipart.call_args.kwargs["data"]
        assert data.get("language") == "pt"

    async def test_per_call_kwargs_forwarded(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.atranscribe_response(_DUMMY_AUDIO, language="ja")
        data = mock_http.apost_multipart.call_args.kwargs["data"]
        assert data.get("language") == "ja"


class TestTranscribe:
    def test_json_format_returns_transcription_response(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "Hello"})
        result = client.transcribe(_DUMMY_AUDIO)
        assert isinstance(result, TranscriptionResponse)
        assert result.text == "Hello"

    def test_verbose_json_returns_transcription_response_with_extras(self):
        client, mock_http = _make_stt(response_format="verbose_json")
        mock_http.post_multipart.return_value = _make_httpx_json_response(
            {"text": "Hello", "language": "en", "duration": 1.5}
        )
        result = client.transcribe(_DUMMY_AUDIO)
        assert isinstance(result, TranscriptionResponse)
        assert result.model_extra["language"] == "en"

    def test_text_format_returns_str(self):
        client, mock_http = _make_stt(response_format="text")
        mock_http.post_multipart.return_value = _make_httpx_text_response("Hello world")
        result = client.transcribe(_DUMMY_AUDIO)
        assert isinstance(result, str)
        assert result == "Hello world"

    def test_srt_format_returns_str(self):
        srt = "1\n00:00:00,000 --> 00:00:02,000\nHello\n"
        client, mock_http = _make_stt(response_format="srt")
        mock_http.post_multipart.return_value = _make_httpx_text_response(srt)
        result = client.transcribe(_DUMMY_AUDIO)
        assert isinstance(result, str)

    def test_vtt_format_returns_str(self):
        vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello\n"
        client, mock_http = _make_stt(response_format="vtt")
        mock_http.post_multipart.return_value = _make_httpx_text_response(vtt)
        result = client.transcribe(_DUMMY_AUDIO)
        assert isinstance(result, str)

    def test_per_call_response_format_overrides_instance_format(self):
        client, mock_http = _make_stt(response_format="json")
        mock_http.post_multipart.return_value = _make_httpx_text_response("plain text")
        result = client.transcribe(_DUMMY_AUDIO, response_format="text")
        assert isinstance(result, str)

    def test_audio_bytes_forwarded_to_http(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.transcribe(_DUMMY_AUDIO)
        files = mock_http.post_multipart.call_args.kwargs["files"]
        _, content, _ = files["file"]
        assert content is _DUMMY_AUDIO

    def test_file_name_kwarg_changes_filename_and_mime(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.transcribe(_DUMMY_AUDIO, file_name="speech.wav")
        files = mock_http.post_multipart.call_args.kwargs["files"]
        filename, _, mime = files["file"]
        assert filename == "speech.wav"
        assert mime == "audio/wav"

    def test_json_result_text_field_correct(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response(
            {"text": "Precise transcription"}
        )
        result = client.transcribe(_DUMMY_AUDIO)
        assert result.text == "Precise transcription"


@pytest.mark.asyncio
class TestATranscribe:
    async def test_json_format_returns_transcription_response(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "Hello"})
        result = await client.atranscribe(_DUMMY_AUDIO)
        assert isinstance(result, TranscriptionResponse)
        assert result.text == "Hello"

    async def test_text_format_returns_str(self):
        client, mock_http = _make_stt(response_format="text")
        mock_http.apost_multipart.return_value = _make_httpx_text_response("Hello world")
        result = await client.atranscribe(_DUMMY_AUDIO)
        assert isinstance(result, str)
        assert result == "Hello world"

    async def test_verbose_json_extra_fields_accessible(self):
        client, mock_http = _make_stt(response_format="verbose_json")
        mock_http.apost_multipart.return_value = _make_httpx_json_response(
            {"text": "Hi", "language": "en", "segments": []}
        )
        result = await client.atranscribe(_DUMMY_AUDIO)
        assert isinstance(result, TranscriptionResponse)
        assert result.model_extra["language"] == "en"

    async def test_srt_format_returns_str(self):
        srt = "1\n00:00:00,000 --> 00:00:02,000\nHello\n"
        client, mock_http = _make_stt(response_format="srt")
        mock_http.apost_multipart.return_value = _make_httpx_text_response(srt)
        result = await client.atranscribe(_DUMMY_AUDIO)
        assert isinstance(result, str)

    async def test_vtt_format_returns_str(self):
        vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello\n"
        client, mock_http = _make_stt(response_format="vtt")
        mock_http.apost_multipart.return_value = _make_httpx_text_response(vtt)
        result = await client.atranscribe(_DUMMY_AUDIO)
        assert isinstance(result, str)

    async def test_audio_bytes_forwarded(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.atranscribe(_DUMMY_AUDIO)
        files = mock_http.apost_multipart.call_args.kwargs["files"]
        _, content, _ = files["file"]
        assert content is _DUMMY_AUDIO

    async def test_file_name_kwarg_forwarded(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.atranscribe(_DUMMY_AUDIO, file_name="audio.webm")
        files = mock_http.apost_multipart.call_args.kwargs["files"]
        filename, _, mime = files["file"]
        assert filename == "audio.webm"
        assert mime == "audio/webm"

    async def test_per_call_response_format_overrides_instance(self):
        client, mock_http = _make_stt(response_format="json")
        mock_http.apost_multipart.return_value = _make_httpx_text_response("plain text")
        result = await client.atranscribe(_DUMMY_AUDIO, response_format="text")
        assert isinstance(result, str)


class TestInvoke:
    def test_returns_transcription_response_for_json_format(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "Hello"})
        result = client.invoke(_DUMMY_AUDIO)
        assert isinstance(result, TranscriptionResponse)
        assert result.text == "Hello"

    def test_config_none_accepted(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "Hello"})
        result = client.invoke(_DUMMY_AUDIO, config=None)
        assert isinstance(result, TranscriptionResponse)

    def test_config_object_is_ignored(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "Hello"})
        result = client.invoke(_DUMMY_AUDIO, config={"callbacks": [], "tags": ["test"]})
        assert isinstance(result, TranscriptionResponse)

    def test_kwargs_forwarded_to_transcribe(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_text_response("Hello")
        result = client.invoke(_DUMMY_AUDIO, response_format="text")
        assert isinstance(result, str)

    def test_audio_bytes_forwarded_to_http(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.invoke(_DUMMY_AUDIO)
        files = mock_http.post_multipart.call_args.kwargs["files"]
        _, content, _ = files["file"]
        assert content is _DUMMY_AUDIO

    def test_file_name_kwarg_forwarded(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.invoke(_DUMMY_AUDIO, file_name="grabacion.wav")
        files = mock_http.post_multipart.call_args.kwargs["files"]
        filename, _, mime = files["file"]
        assert filename == "grabacion.wav"
        assert mime == "audio/wav"

    def test_post_multipart_called_once(self, stt_client):
        client, mock_http = stt_client
        mock_http.post_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        client.invoke(_DUMMY_AUDIO)
        mock_http.post_multipart.assert_called_once()


@pytest.mark.asyncio
class TestAInvoke:
    async def test_returns_transcription_response_for_json_format(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "Hello"})
        result = await client.ainvoke(_DUMMY_AUDIO)
        assert isinstance(result, TranscriptionResponse)
        assert result.text == "Hello"

    async def test_config_none_accepted(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "Hello"})
        result = await client.ainvoke(_DUMMY_AUDIO, config=None)
        assert isinstance(result, TranscriptionResponse)

    async def test_config_object_is_ignored(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "Hello"})
        result = await client.ainvoke(_DUMMY_AUDIO, config={"metadata": {}})
        assert isinstance(result, TranscriptionResponse)

    async def test_kwargs_forwarded_to_atranscribe(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_text_response("Hello")
        result = await client.ainvoke(_DUMMY_AUDIO, response_format="text")
        assert isinstance(result, str)

    async def test_audio_bytes_forwarded_to_http(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.ainvoke(_DUMMY_AUDIO)
        files = mock_http.apost_multipart.call_args.kwargs["files"]
        _, content, _ = files["file"]
        assert content is _DUMMY_AUDIO

    async def test_file_name_kwarg_forwarded(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.ainvoke(_DUMMY_AUDIO, file_name="clip.m4a")
        files = mock_http.apost_multipart.call_args.kwargs["files"]
        filename, _, mime = files["file"]
        assert filename == "clip.m4a"
        assert mime == "audio/mp4"

    async def test_apost_multipart_called_once(self, stt_client):
        client, mock_http = stt_client
        mock_http.apost_multipart.return_value = _make_httpx_json_response({"text": "hi"})
        await client.ainvoke(_DUMMY_AUDIO)
        mock_http.apost_multipart.assert_called_once()
