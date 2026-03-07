from __future__ import annotations

import json
import warnings
from typing import Any

import httpx
import pytest
from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError

load_dotenv(find_dotenv())

from langchain_pollinations._audio_catalog import (  # noqa: E402
    AudioModelId,
    _FALLBACK_AUDIO_MODEL_IDS,
    _audio_model_ids_cache,
    _audio_model_ids_loaded,
    _audio_model_ids_lock,
    _load_audio_model_ids,
)
from langchain_pollinations._errors import PollinationsAPIError  # noqa: E402
from langchain_pollinations.stt import (  # noqa: E402
    _AUDIO_MIME_TYPES,
    _FALLBACK_MIME_TYPE,
    STTPollinations,
    TranscriptionParams,
    TranscriptionResponse,
)
from langchain_pollinations.tts import TTSPollinations  # noqa: E402

_TRANSCRIPTION_TEXT = "The quick brown fox jumps over the lazy dog."
_SIMPLE_TEXT = "Hello. Integration test."

# Modelos STT válidos para tests parametrizados (según la spec del API).
# "suno" es un modelo inválido; tiene su propio test de warning.
_VALID_STT_MODELS = [
    pytest.param("whisper-large-v3", id="whisper"),
    pytest.param("scribe", id="scribe"),
]
_API_BUG_NON_JSON_FORMAT = (
    "Pollinations API v0.3.0 returns HTTP 500 for non-JSON response_format values "
    "(text, srt, vtt). The backend tries to JSON.parse() the plain-text transcription "
    "result from Whisper and crashes. Spec documents only application/json responses."
)


def _fake_response(
    status_code: int,
    body: dict[str, Any] | str,
    content_type: str = "application/json",
) -> httpx.Response:
    """
    Build a minimal :class:`httpx.Response` suitable for unit-level parsing
    tests. Only ``status_code``, ``Content-Type``, and ``content`` are set.
    """
    if isinstance(body, dict):
        raw = json.dumps(body).encode()
    elif isinstance(body, str):
        raw = body.encode()
    else:
        raw = body
    return httpx.Response(status_code, headers={"content-type": content_type}, content=raw)


@pytest.fixture(scope="session")
def api_key() -> str:
    """Resolve the API key from the loaded .env; skip the whole session if absent."""
    import os

    key = os.environ.get("POLLINATIONS_API_KEY", "")
    if not key:
        pytest.skip("POLLINATIONS_API_KEY not set — integration tests skipped.")
    return key


@pytest.fixture(scope="session")
def tts_client(api_key: str) -> TTSPollinations:
    """Session-scoped TTS client used to produce audio for STT tests."""
    return TTSPollinations(api_key=api_key, voice="alloy", response_format="mp3")


@pytest.fixture(scope="session")
def sample_audio_mp3(tts_client: TTSPollinations) -> bytes:
    """
    MP3 audio bytes from a short, clearly intelligible English sentence.
    Generated once per session to minimize API calls.
    """
    return tts_client.generate(_TRANSCRIPTION_TEXT)


@pytest.fixture(scope="session")
def sample_audio_wav() -> bytes:
    """
    Minimal valid WAV bytes generated with Python's standard ``wave`` module.

    Uses 1 second of silence at 16 kHz / 16-bit mono — the canonical format
    accepted by all Whisper-compatible STT endpoints. Generated locally so
    this fixture has no dependency on the TTS API and produces well-formed
    WAV headers that the upstream Whisper service can parse correctly.

    TTS-generated WAV files are unsuitable here because they may carry
    non-standard headers or sample rates that cause upstream duration
    miscalculation (observed: HTTP 413 from Whisper reporting multi-hour
    duration for a 2-second clip).
    """
    import io
    import wave

    sample_rate = 16_000  # Hz, estándar de Whisper
    duration_s = 1
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)                          # mono
        wf.setsampwidth(2)                          # 16-bit PCM
        wf.setframerate(sample_rate)
        # 1 segundo de silencio; el STT devuelve texto vacío o mínimo.
        wf.writeframes(b"\x00\x00" * (sample_rate * duration_s))
    return buf.getvalue()


@pytest.fixture
def stt(api_key: str) -> STTPollinations:
    """Default :class:`STTPollinations` instance with no extra configuration."""
    return STTPollinations(api_key=api_key)


class TestAudioCatalog:
    def test_fallback_list_is_non_empty(self) -> None:
        assert len(_FALLBACK_AUDIO_MODEL_IDS) > 0

    def test_fallback_contains_whisper_large_v3(self) -> None:
        assert "whisper-large-v3" in _FALLBACK_AUDIO_MODEL_IDS

    def test_fallback_contains_whisper_1(self) -> None:
        assert "whisper-1" in _FALLBACK_AUDIO_MODEL_IDS

    def test_fallback_contains_scribe(self) -> None:
        assert "scribe" in _FALLBACK_AUDIO_MODEL_IDS

    def test_audio_model_id_type_alias_is_str(self) -> None:
        assert AudioModelId is str

    def test_cache_initialised_from_fallback(self) -> None:
        assert isinstance(_audio_model_ids_cache, list)
        assert len(_audio_model_ids_cache) > 0

    def test_lock_is_threading_lock(self) -> None:
        import threading

        assert isinstance(_audio_model_ids_lock, type(threading.Lock()))

    def test_loaded_flag_is_bool(self) -> None:
        assert isinstance(_audio_model_ids_loaded, bool)

    def test_load_returns_list(self, api_key: str) -> None:
        ids = _load_audio_model_ids(api_key, force=True)
        assert isinstance(ids, list)

    def test_load_all_elements_are_strings(self, api_key: str) -> None:
        ids = _load_audio_model_ids(api_key)
        assert all(isinstance(i, str) for i in ids)

    def test_load_list_is_non_empty(self, api_key: str) -> None:
        ids = _load_audio_model_ids(api_key)
        assert len(ids) > 0

    def test_load_second_call_returns_same_list(self, api_key: str) -> None:
        first = _load_audio_model_ids(api_key)
        second = _load_audio_model_ids(api_key)
        assert first == second

    def test_load_force_refreshes_cache(self, api_key: str) -> None:
        ids = _load_audio_model_ids(api_key, force=True)
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_load_returns_independent_copy(self, api_key: str) -> None:
        """Mutations of the returned list must not affect the module-level cache."""
        ids = _load_audio_model_ids(api_key)
        original_len = len(ids)
        ids.append("__test_sentinel__")
        fresh = _load_audio_model_ids(api_key)
        assert "__test_sentinel__" not in fresh
        assert len(fresh) == original_len

    def test_load_without_api_key_does_not_raise(self) -> None:
        """_load_audio_model_ids must fail silently when no key is available."""
        try:
            result = _load_audio_model_ids(None, force=True)
            assert isinstance(result, list)
        except Exception as exc:
            pytest.fail(f"_load_audio_model_ids raised unexpectedly: {exc}")


class TestTranscriptionParams:
    def test_default_model(self) -> None:
        assert TranscriptionParams().model == "whisper-large-v3"

    def test_default_response_format(self) -> None:
        assert TranscriptionParams().response_format == "json"

    def test_default_optionals_are_none(self) -> None:
        p = TranscriptionParams()
        assert p.language is None
        assert p.prompt is None
        assert p.temperature is None

    def test_to_form_data_excludes_none_fields(self) -> None:
        data = TranscriptionParams().to_form_data()
        assert "language" not in data
        assert "prompt" not in data
        assert "temperature" not in data

    def test_to_form_data_includes_model(self) -> None:
        data = TranscriptionParams().to_form_data()
        assert "model" in data
        assert data["model"] == "whisper-large-v3"

    def test_to_form_data_all_values_are_strings(self) -> None:
        p = TranscriptionParams(temperature=0.5, language="en")
        data = p.to_form_data()
        assert all(isinstance(v, str) for v in data.values())

    def test_to_form_data_temperature_cast_to_string(self) -> None:
        p = TranscriptionParams(temperature=0.3)
        assert p.to_form_data()["temperature"] == "0.3"

    def test_to_form_data_all_fields_populated(self) -> None:
        p = TranscriptionParams(
            model="scribe",
            language="es",
            prompt="Contexto de prueba.",
            response_format="verbose_json",
            temperature=0.1,
        )
        data = p.to_form_data()
        assert data["model"] == "scribe"
        assert data["language"] == "es"
        assert data["prompt"] == "Contexto de prueba."
        assert data["response_format"] == "verbose_json"
        assert data["temperature"] == "0.1"

    def test_temperature_lower_bound_accepted(self) -> None:
        assert TranscriptionParams(temperature=0.0).temperature == 0.0

    def test_temperature_upper_bound_accepted(self) -> None:
        assert TranscriptionParams(temperature=1.0).temperature == 1.0

    def test_temperature_below_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptionParams(temperature=-0.01)

    def test_temperature_above_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptionParams(temperature=1.01)

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptionParams(nonexistent_field="value")

    @pytest.mark.parametrize("fmt", ["json", "text", "srt", "verbose_json", "vtt"])
    def test_valid_response_formats(self, fmt: str) -> None:
        assert TranscriptionParams(response_format=fmt).response_format == fmt  # type: ignore[arg-type]

    def test_invalid_response_format_raises(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptionParams(response_format="xml")  # type: ignore[arg-type]


class TestTranscriptionResponse:
    def test_text_field_stored(self) -> None:
        r = TranscriptionResponse(text="hello world")
        assert r.text == "hello world"

    def test_text_is_required(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptionResponse()  # type: ignore[call-arg]

    def test_extra_fields_accepted(self) -> None:
        r = TranscriptionResponse(text="hi", language="en", duration=2.5)
        assert r.model_extra["language"] == "en"
        assert r.model_extra["duration"] == 2.5

    def test_verbose_json_segments_preserved(self) -> None:
        r = TranscriptionResponse(
            text="hi",
            segments=[{"id": 0, "text": "hi", "start": 0.0, "end": 0.5}],
        )
        assert len(r.model_extra["segments"]) == 1

    def test_model_validate_from_dict(self) -> None:
        r = TranscriptionResponse.model_validate({"text": "test"})
        assert r.text == "test"

    def test_model_validate_verbose_json_payload(self) -> None:
        payload = {
            "text": "hello",
            "language": "en",
            "duration": 1.2,
            "task": "transcribe",
            "segments": [],
        }
        r = TranscriptionResponse.model_validate(payload)
        assert r.text == "hello"
        assert r.model_extra["language"] == "en"

    def test_empty_text_accepted(self) -> None:
        r = TranscriptionResponse(text="")
        assert r.text == ""


class TestSTTPollinationsConfig:
    def test_default_base_url(self, api_key: str) -> None:
        assert STTPollinations(api_key=api_key).base_url == "https://gen.pollinations.ai"

    def test_default_timeout(self, api_key: str) -> None:
        assert STTPollinations(api_key=api_key).timeout_s == 120.0

    def test_default_fields_are_none(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key)
        for field in ("model", "language", "prompt", "response_format", "temperature"):
            assert getattr(s, field) is None

    def test_default_file_name(self, api_key: str) -> None:
        assert STTPollinations(api_key=api_key).file_name == "audio.mp3"

    def test_init_with_all_fields(self, api_key: str) -> None:
        s = STTPollinations(
            api_key=api_key,
            model="scribe",
            language="es",
            prompt="contexto",
            response_format="verbose_json",
            temperature=0.2,
            file_name="audio.wav",
        )
        assert s.model == "scribe"
        assert s.language == "es"
        assert s.prompt == "contexto"
        assert s.response_format == "verbose_json"
        assert s.temperature == 0.2
        assert s.file_name == "audio.wav"

    def test_api_key_excluded_from_repr(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key)
        assert api_key not in repr(s)

    def test_unknown_model_emits_user_warning(self, api_key: str) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            STTPollinations(api_key=api_key, model="nonexistent-model-xyz")
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warns) > 0
        assert "nonexistent-model-xyz" in str(user_warns[0].message)

    def test_known_model_emits_no_warning(self, api_key: str) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            STTPollinations(api_key=api_key, model="whisper-large-v3")
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warns) == 0

    def test_suno_model_emits_user_warning(self, api_key: str) -> None:
        """'suno' no es un modelo STT conocido; se espera UserWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            STTPollinations(api_key=api_key, model="suno")
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warns) > 0
        assert "suno" in str(user_warns[0].message)

    def test_temperature_out_of_range_raises(self, api_key: str) -> None:
        with pytest.raises(ValidationError):
            STTPollinations(api_key=api_key, temperature=1.5)

    def test_extra_field_raises(self, api_key: str) -> None:
        with pytest.raises(ValidationError):
            STTPollinations(api_key=api_key, nonexistent_param="x")

    def test_defaults_dict_empty_when_all_none(self, api_key: str) -> None:
        assert STTPollinations(api_key=api_key)._defaults_dict() == {}

    def test_defaults_dict_populated(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key, model="scribe", language="fr", response_format="text")
        d = s._defaults_dict()
        assert d["model"] == "scribe"
        assert d["language"] == "fr"
        assert d["response_format"] == "text"
        assert "temperature" not in d

    def test_defaults_dict_excludes_file_name(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key, file_name="clip.wav")
        assert "file_name" not in s._defaults_dict()

    def test_with_params_returns_new_instance(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key)
        updated = s.with_params(language="de")
        assert updated is not s

    def test_with_params_does_not_mutate_original(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key)
        s.with_params(language="de")
        assert s.language is None

    def test_with_params_overrides_field(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key, language="en")
        assert s.with_params(language="ja").language == "ja"

    def test_with_params_preserves_api_key(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key)
        assert s.with_params(language="it").api_key == api_key

    def test_with_params_sets_multiple_fields(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key)
        updated = s.with_params(language="de", response_format="text", temperature=0.3)
        assert updated.language == "de"
        assert updated.response_format == "text"
        assert updated.temperature == 0.3


class TestMimeTypeFor:
    @pytest.mark.parametrize(
        "filename, expected_mime",
        [
            ("audio.mp3", "audio/mpeg"),
            ("audio.mp4", "audio/mp4"),
            ("audio.mpeg", "audio/mpeg"),
            ("audio.mpga", "audio/mpeg"),
            ("audio.m4a", "audio/mp4"),
            ("audio.wav", "audio/wav"),
            ("audio.webm", "audio/webm"),
        ],
    )
    def test_known_extension(self, filename: str, expected_mime: str) -> None:
        assert STTPollinations._mime_type_for(filename) == expected_mime

    def test_uppercase_extension_normalised(self) -> None:
        assert STTPollinations._mime_type_for("audio.MP3") == "audio/mpeg"

    def test_mixed_case_extension_normalised(self) -> None:
        assert STTPollinations._mime_type_for("audio.Wav") == "audio/wav"

    def test_unknown_extension_returns_fallback(self) -> None:
        assert STTPollinations._mime_type_for("audio.ogg") == _FALLBACK_MIME_TYPE

    def test_no_extension_returns_fallback(self) -> None:
        assert STTPollinations._mime_type_for("audiofile") == _FALLBACK_MIME_TYPE

    def test_mime_types_dict_fully_reachable(self) -> None:
        """Every entry in _AUDIO_MIME_TYPES must be reachable via _mime_type_for."""
        for ext, expected in _AUDIO_MIME_TYPES.items():
            assert STTPollinations._mime_type_for(f"file.{ext}") == expected


_DUMMY_AUDIO = b"\xff\xfb\x90\x00" * 64  # bytes mínimos que simulan un MP3


class TestBuildMultipart:
    def test_returns_three_tuple(self, api_key: str) -> None:
        result = STTPollinations(api_key=api_key)._build_multipart(_DUMMY_AUDIO)
        assert len(result) == 3

    def test_files_dict_has_file_key(self, api_key: str) -> None:
        files_dict, _, _ = STTPollinations(api_key=api_key)._build_multipart(_DUMMY_AUDIO)
        assert "file" in files_dict

    def test_files_dict_tuple_has_three_elements(self, api_key: str) -> None:
        files_dict, _, _ = STTPollinations(api_key=api_key)._build_multipart(_DUMMY_AUDIO)
        assert len(files_dict["file"]) == 3

    def test_files_dict_content_is_original_bytes(self, api_key: str) -> None:
        files_dict, _, _ = STTPollinations(api_key=api_key)._build_multipart(_DUMMY_AUDIO)
        assert files_dict["file"][1] is _DUMMY_AUDIO

    def test_default_file_name_used(self, api_key: str) -> None:
        files_dict, _, _ = STTPollinations(api_key=api_key)._build_multipart(_DUMMY_AUDIO)
        assert files_dict["file"][0] == "audio.mp3"

    def test_instance_file_name_used(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key, file_name="clip.wav")
        files_dict, _, _ = s._build_multipart(_DUMMY_AUDIO)
        assert files_dict["file"][0] == "clip.wav"
        assert files_dict["file"][2] == "audio/wav"

    def test_file_name_override_via_kwargs(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key)
        files_dict, _, _ = s._build_multipart(_DUMMY_AUDIO, file_name="override.wav")
        assert files_dict["file"][0] == "override.wav"
        assert files_dict["file"][2] == "audio/wav"

    def test_file_name_override_via_params(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key)
        files_dict, _, _ = s._build_multipart(_DUMMY_AUDIO, params={"file_name": "via_params.m4a"})
        assert files_dict["file"][0] == "via_params.m4a"
        assert files_dict["file"][2] == "audio/mp4"

    def test_form_data_all_values_are_strings(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key, temperature=0.4)
        _, form_data, _ = s._build_multipart(_DUMMY_AUDIO)
        assert all(isinstance(v, str) for v in form_data.values())

    def test_form_data_excludes_file_name(self, api_key: str) -> None:
        _, form_data, _ = STTPollinations(api_key=api_key)._build_multipart(_DUMMY_AUDIO)
        assert "file_name" not in form_data

    def test_params_obj_response_format_default(self, api_key: str) -> None:
        _, _, params_obj = STTPollinations(api_key=api_key)._build_multipart(_DUMMY_AUDIO)
        assert params_obj.response_format == "json"

    def test_params_obj_response_format_from_instance(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key, response_format="text")
        _, _, params_obj = s._build_multipart(_DUMMY_AUDIO)
        assert params_obj.response_format == "text"

    def test_instance_defaults_applied(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key, language="en", model="scribe")
        _, _, params_obj = s._build_multipart(_DUMMY_AUDIO)
        assert params_obj.language == "en"
        assert params_obj.model == "scribe"

    def test_params_dict_overrides_instance_defaults(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key, model="whisper-large-v3")
        _, _, params_obj = s._build_multipart(_DUMMY_AUDIO, params={"model": "scribe"})
        assert params_obj.model == "scribe"

    def test_kwargs_override_instance_defaults(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key, language="en")
        _, _, params_obj = s._build_multipart(_DUMMY_AUDIO, language="fr")
        assert params_obj.language == "fr"

    def test_kwargs_override_params_dict(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key)
        _, _, params_obj = s._build_multipart(
            _DUMMY_AUDIO, params={"language": "en"}, language="ja"
        )
        assert params_obj.language == "ja"

    def test_invalid_temperature_raises_validation_error(self, api_key: str) -> None:
        s = STTPollinations(api_key=api_key)
        with pytest.raises(ValidationError):
            s._build_multipart(_DUMMY_AUDIO, temperature=5.0)


class TestParseTranscriptionResponse:
    def test_json_content_type_returns_transcription_response(self) -> None:
        resp = _fake_response(200, {"text": "hello world"})
        # content-type es application/json → siempre TranscriptionResponse
        result = STTPollinations._parse_transcription_response(resp, "json")
        assert isinstance(result, TranscriptionResponse)
        assert result.text == "hello world"

    def test_verbose_json_content_type_returns_transcription_response(self) -> None:
        resp = _fake_response(200, {"text": "hello", "language": "en", "duration": 1.0})
        result = STTPollinations._parse_transcription_response(resp, "verbose_json")
        assert isinstance(result, TranscriptionResponse)
        assert result.model_extra["language"] == "en"

    def test_json_content_type_with_text_format_param_still_returns_response(self) -> None:
        """
        Aunque response_format='text' fue solicitado, si el API devuelve
        application/json (comportamiento actual de Pollinations), se parsea como JSON.
        """
        resp = _fake_response(200, {"text": "plain result"})
        result = STTPollinations._parse_transcription_response(resp, "text")
        assert isinstance(result, TranscriptionResponse)
        assert result.text == "plain result"

    def test_non_json_content_type_returns_str(self) -> None:
        """
        Si el API algún día devuelve text/plain para response_format=text,
        el método retorna str correctamente.
        """
        resp = _fake_response(200, "plain result", content_type="text/plain")
        result = STTPollinations._parse_transcription_response(resp, "text")
        assert isinstance(result, str)
        assert result == "plain result"

    def test_srt_content_type_returns_str(self) -> None:
        srt = "1\n00:00:00,000 --> 00:00:01,000\nhello\n"
        resp = _fake_response(200, srt, content_type="text/plain")
        result = STTPollinations._parse_transcription_response(resp, "srt")
        assert isinstance(result, str)
        assert "00:00:00" in result

    def test_vtt_content_type_returns_str(self) -> None:
        vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n"
        resp = _fake_response(200, vtt, content_type="text/plain")
        result = STTPollinations._parse_transcription_response(resp, "vtt")
        assert isinstance(result, str)
        assert "WEBVTT" in result

    def test_verbose_json_preserves_extra_fields(self) -> None:
        payload = {"text": "hi", "task": "transcribe", "segments": [{"id": 0}]}
        resp = _fake_response(200, payload)
        result = STTPollinations._parse_transcription_response(resp, "verbose_json")
        assert isinstance(result, TranscriptionResponse)
        assert result.model_extra["task"] == "transcribe"


class TestTranscribeResponse:
    def test_returns_httpx_response(self, stt: STTPollinations, sample_audio_mp3: bytes) -> None:
        resp = stt.transcribe_response(sample_audio_mp3)
        assert isinstance(resp, httpx.Response)

    def test_status_code_200(self, stt: STTPollinations, sample_audio_mp3: bytes) -> None:
        resp = stt.transcribe_response(sample_audio_mp3)
        assert resp.status_code == 200

    def test_body_contains_text_key(self, stt: STTPollinations, sample_audio_mp3: bytes) -> None:
        body = stt.transcribe_response(sample_audio_mp3).json()
        assert "text" in body

    @pytest.mark.asyncio
    async def test_async_returns_httpx_response(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        resp = await stt.atranscribe_response(sample_audio_mp3)
        assert isinstance(resp, httpx.Response)
        assert resp.status_code == 200


class TestTranscribeJson:
    def test_returns_transcription_response(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        result = stt.transcribe(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)

    def test_text_is_non_empty_string(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        result = stt.transcribe(sample_audio_mp3)
        assert isinstance(result.text, str)
        assert result.text.strip() != ""

    @pytest.mark.parametrize("model", _VALID_STT_MODELS)
    def test_transcribe_per_model(
        self, model: str, api_key: str, sample_audio_mp3: bytes
    ) -> None:
        s = STTPollinations(api_key=api_key, model=model)
        result = s.transcribe(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)
        assert len(result.text) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", _VALID_STT_MODELS)
    async def test_atranscribe_per_model(
        self, model: str, api_key: str, sample_audio_mp3: bytes
    ) -> None:
        s = STTPollinations(api_key=api_key, model=model)
        result = await s.atranscribe(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)
        assert len(result.text) > 0


class TestTranscribeFormats:
    def test_format_json_returns_transcription_response(
        self, api_key: str, sample_audio_mp3: bytes
    ) -> None:
        s = STTPollinations(api_key=api_key, response_format="json")
        result = s.transcribe(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)
        assert len(result.text) > 0

    @pytest.mark.xfail(
        raises=PollinationsAPIError,
        strict=True,
        reason=_API_BUG_NON_JSON_FORMAT,
    )
    def test_format_text_returns_str(self, api_key: str, sample_audio_mp3: bytes) -> None:
        s = STTPollinations(api_key=api_key, response_format="text")
        result = s.transcribe(sample_audio_mp3)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.xfail(
        raises=PollinationsAPIError,
        strict=True,
        reason=_API_BUG_NON_JSON_FORMAT,
    )
    def test_format_srt_returns_str(self, api_key: str, sample_audio_mp3: bytes) -> None:
        s = STTPollinations(api_key=api_key, response_format="srt")
        result = s.transcribe(sample_audio_mp3)
        assert isinstance(result, str)

    @pytest.mark.xfail(
        raises=PollinationsAPIError,
        strict=True,
        reason=_API_BUG_NON_JSON_FORMAT,
    )
    def test_format_vtt_returns_str(self, api_key: str, sample_audio_mp3: bytes) -> None:
        s = STTPollinations(api_key=api_key, response_format="vtt")
        result = s.transcribe(sample_audio_mp3)
        assert isinstance(result, str)

    def test_format_verbose_json_returns_transcription_response(
        self, api_key: str, sample_audio_mp3: bytes
    ) -> None:
        s = STTPollinations(api_key=api_key, response_format="verbose_json")
        result = s.transcribe(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)
        assert len(result.text) > 0

    def test_format_override_per_call(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        # Override a JSON-compatible format para evitar el bug del servidor.
        result = stt.transcribe(sample_audio_mp3, response_format="verbose_json")
        assert isinstance(result, TranscriptionResponse)

    def test_format_override_via_params_dict(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        result = stt.transcribe(sample_audio_mp3, params={"response_format": "verbose_json"})
        assert isinstance(result, TranscriptionResponse)

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        raises=PollinationsAPIError,
        strict=True,
        reason=_API_BUG_NON_JSON_FORMAT,
    )
    async def test_async_format_text(self, api_key: str, sample_audio_mp3: bytes) -> None:
        s = STTPollinations(api_key=api_key, response_format="text")
        result = await s.atranscribe(sample_audio_mp3)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_async_format_verbose_json(
        self, api_key: str, sample_audio_mp3: bytes
    ) -> None:
        s = STTPollinations(api_key=api_key, response_format="verbose_json")
        result = await s.atranscribe(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)


class TestTranscribeCallParams:
    def test_language_hint(self, api_key: str, sample_audio_mp3: bytes) -> None:
        s = STTPollinations(api_key=api_key, language="en")
        result = s.transcribe(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)

    def test_prompt_context(self, api_key: str, sample_audio_mp3: bytes) -> None:
        s = STTPollinations(api_key=api_key, prompt="A sentence about a fox.")
        result = s.transcribe(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)

    def test_temperature_zero(self, api_key: str, sample_audio_mp3: bytes) -> None:
        s = STTPollinations(api_key=api_key, temperature=0.0)
        result = s.transcribe(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)

    def test_file_name_override_kwarg(self, stt: STTPollinations, sample_audio_mp3: bytes) -> None:
        result = stt.transcribe(sample_audio_mp3, file_name="speech.mp3")
        assert isinstance(result, TranscriptionResponse)

    def test_wav_audio_input(self, api_key: str, sample_audio_wav: bytes) -> None:
        s = STTPollinations(api_key=api_key, file_name="audio.wav")
        result = s.transcribe(sample_audio_wav)
        # El audio es silencio; text puede ser "" o una cadena mínima.
        # Lo que se verifica es que el API acepta WAV correctamente formado
        # y responde con TranscriptionResponse sin error.
        assert isinstance(result, TranscriptionResponse)
        assert isinstance(result.text, str)

    def test_params_dict_applied(self, stt: STTPollinations, sample_audio_mp3: bytes) -> None:
        result = stt.transcribe(sample_audio_mp3, params={"language": "en"})
        assert isinstance(result, TranscriptionResponse)

    def test_kwargs_priority_over_params_dict(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        # response_format="verbose_json" via kwargs debe ganar sobre "text" via params
        result = stt.transcribe(
            sample_audio_mp3,
            params={"response_format": "text"},
            response_format="verbose_json",
        )
        assert isinstance(result, TranscriptionResponse)

    def test_with_params_affects_real_request(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        verbose_stt = stt.with_params(response_format="verbose_json")
        result = verbose_stt.transcribe(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)


class TestInvokeInterface:
    def test_invoke_returns_transcription_response(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        result = stt.invoke(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)
        assert len(result.text) > 0

    def test_invoke_config_none_accepted(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        result = stt.invoke(sample_audio_mp3, config=None)
        assert isinstance(result, TranscriptionResponse)

    def test_invoke_config_dict_accepted(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        result = stt.invoke(sample_audio_mp3, config={"run_name": "test"})
        assert isinstance(result, TranscriptionResponse)

    @pytest.mark.xfail(
        raises=PollinationsAPIError,
        strict=True,
        reason=_API_BUG_NON_JSON_FORMAT,
    )
    def test_invoke_passes_kwargs_to_transcribe(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        result = stt.invoke(sample_audio_mp3, response_format="text")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_ainvoke_returns_transcription_response(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        result = await stt.ainvoke(sample_audio_mp3)
        assert isinstance(result, TranscriptionResponse)
        assert len(result.text) > 0

    @pytest.mark.asyncio
    async def test_ainvoke_config_none_accepted(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        result = await stt.ainvoke(sample_audio_mp3, config=None)
        assert isinstance(result, TranscriptionResponse)

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        raises=PollinationsAPIError,
        strict=True,
        reason=_API_BUG_NON_JSON_FORMAT,
    )
    async def test_ainvoke_passes_kwargs(
        self, stt: STTPollinations, sample_audio_mp3: bytes
    ) -> None:
        result = await stt.ainvoke(sample_audio_mp3, response_format="text")
        assert isinstance(result, str)


class TestTTSToSTTRoundtrip:
    def test_sync_roundtrip_default_models(
        self, tts_client: TTSPollinations, stt: STTPollinations
    ) -> None:
        audio = tts_client.generate("Integration roundtrip test.")
        result = stt.transcribe(audio)
        assert isinstance(result, TranscriptionResponse)
        assert len(result.text) > 0

    @pytest.mark.parametrize("model", _VALID_STT_MODELS)
    def test_sync_roundtrip_per_stt_model(
        self, model: str, tts_client: TTSPollinations, api_key: str
    ) -> None:
        audio = tts_client.generate("Testing model selection.")
        s = STTPollinations(api_key=api_key, model=model)
        result = s.transcribe(audio)
        assert isinstance(result, TranscriptionResponse)
        assert len(result.text) > 0

    def test_sync_roundtrip_suno_model_warns_then_proceeds(
        self, tts_client: TTSPollinations, api_key: str
    ) -> None:
        """
        'suno' no es un modelo STT conocido. Se espera:
        1. UserWarning en la construcción de STTPollinations.
        2. La request se envía de todos modos (el test no asume éxito ni fallo del API).
        """
        audio = tts_client.generate("Suno model test.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            s = STTPollinations(api_key=api_key, model="suno")
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warns) > 0
        assert "suno" in str(user_warns[0].message)
        # La llamada puede fallar con PollinationsAPIError si el API rechaza el modelo;
        # lo importante es que el cliente no lanza excepción en construcción.
        try:
            stt_result = s.transcribe(audio)
            assert isinstance(stt_result, (TranscriptionResponse, str))
        except PollinationsAPIError:
            pass  # Comportamiento esperado ante un modelo inválido en el API.

    @pytest.mark.asyncio
    async def test_async_roundtrip(self, api_key: str) -> None:
        tts = TTSPollinations(api_key=api_key, voice="alloy")
        stt = STTPollinations(api_key=api_key)
        audio = tts.generate("Async roundtrip integration test.")
        result = await stt.atranscribe(audio)
        assert isinstance(result, TranscriptionResponse)
        assert len(result.text) > 0

    @pytest.mark.asyncio
    async def test_async_roundtrip_verbose_json(self, api_key: str) -> None:
        tts = TTSPollinations(api_key=api_key, voice="alloy")
        stt = STTPollinations(api_key=api_key, response_format="verbose_json")
        audio = tts.generate("Verbose JSON roundtrip.")
        result = await stt.atranscribe(audio)
        assert isinstance(result, TranscriptionResponse)
        assert len(result.text) > 0

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        raises=PollinationsAPIError,
        strict=True,
        reason=_API_BUG_NON_JSON_FORMAT,
    )
    async def test_async_roundtrip_text_format(self, api_key: str) -> None:
        tts = TTSPollinations(api_key=api_key, voice="alloy")
        stt = STTPollinations(api_key=api_key, response_format="text")
        audio = tts.generate("Text format async roundtrip.")
        result = await stt.atranscribe(audio)
        assert isinstance(result, str)
        assert len(result) > 0




"""
test_wav_audio_input:
Error de generación de wavs:
- Error raíz (upstream): el endpoint interno de Whisper devuelve 413 — Audio duration too long con un 
umbral de 10800 segundos.
- Causa real: el archivo WAV generado por TTSPollinations(response_format="wav") produce un WAV con 
cabeceras o metadata incorrectas que hacen que Whisper calcule mal la duración (e.g., sample rate 
inesperado, bit depth inusual). Es un bug es de parsing del WAV por parte del upstream.
- WAV de baja calidad de formato que TTS genera para reproducción, no para transcripción.

Error de formato:
- Error raíz (servidor): El backend de Pollinations recibe response_format=text, obtiene texto plano 
del modelo Whisper ("The quick..."), e intenta hacer JSON.parse() sobre ese string antes de enviarlo 
al cliente. Eso produce el crash 500 — Unexpected token 'T'. Es un bug del servidor.
- Evidencia en el spec: La sección responses.200 del endpoint /v1/audio/transcriptions documenta una 
sola forma de respuesta: application/json: {"text": string}. El spec acepta response_format como 
parámetro de request pero siempre devuelve JSON. Los formatos text, srt y vtt crashean el gateway al 
intentar wrapear texto plano en JSON.

_parse_transcription_response en stt.py usa response_format del request para decidir cómo parsear la 
respuesta. Pero la respuesta siempre es JSON, así que esa lógica de bifurcación está desalineada con 
la realidad del API.
"""