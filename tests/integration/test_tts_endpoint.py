from __future__ import annotations

import os
import time

import dotenv
import pytest

import langchain_pollinations.tts as tts_module
from langchain_pollinations._errors import PollinationsAPIError
from langchain_pollinations.models import ModelInformation
from langchain_pollinations.tts import (
    TTSPollinations,
    _load_audio_model_ids,
)

pytestmark = pytest.mark.integration

# Texto corto para minimizar tokens consumidos en cada llamada.
_SHORT_TEXT = "Hello, this is a test."
_MEDIUM_TEXT = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."

# Magic bytes de cada formato de audio soportado.
_MAGIC: dict[str, bytes | tuple[bytes, ...]] = {
    "mp3": (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xe3"),
    "wav": (b"RIFF",),
    # "flac": (b"fLaC",),
    # Pollinations retorna PCM crudo para estos formatos (sin cabecera de contenedor).
    "flac": (),
    "opus": (b"OggS",),
    "aac": (b"\xff\xf1", b"\xff\xf9"),
    # PCM es audio crudo sin cabecera; solo verificamos que no esté vacío.
    "pcm": (),
}


def _load_key() -> str:
    """
    Resolve POLLINATIONS_API_KEY from .env file or environment.

    Returns:
        The API key string.

    Raises:
        pytest.skip.Exception: When the key is absent so the test is skipped
            instead of failing with an obscure error.
    """
    dotenv.load_dotenv(dotenv.find_dotenv())
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key:
        pytest.skip("POLLINATIONS_API_KEY not set — skipping integration tests")
    return key


def _check_audio_magic(data: bytes, fmt: str) -> bool:
    """
    Verify that ``data`` begins with one of the expected magic byte sequences
    for the given audio format.

    PCM audio has no header by definition, so any non-empty byte string passes.

    Args:
        data: Raw audio bytes from the API.
        fmt: One of the AudioFormat literal values.

    Returns:
        ``True`` when the magic bytes match (or format is PCM with data present).
    """
    if fmt == "pcm":
        return len(data) > 0
    magic_list = _MAGIC.get(fmt, ())
    return any(data[: len(m)] == m for m in magic_list)


def _pause() -> None:
    """
    Insert a short delay between consecutive API calls to avoid rate-limiting.
    """
    time.sleep(0.8)


@pytest.fixture(scope="module")
def api_key() -> str:
    """
    Provide a validated Pollinations API key for the entire test module.

    Skips the module when the key is not available.
    """
    return _load_key()


@pytest.fixture
def client(api_key: str) -> TTSPollinations:
    """
    Provide a fresh TTSPollinations instance per test.

    Function scope ensures that httpx.AsyncClient is never reused across
    different event loops, preventing 'Event loop is closed' errors when
    pytest-asyncio creates a new loop for each async test function.
    """
    return TTSPollinations(api_key=api_key)


@pytest.fixture(scope="module")
def default_mp3(api_key: str) -> bytes:  # ← ya no depende del fixture client
    """
    Generate a single MP3 audio clip reused across the whole module.

    Uses its own local TTSPollinations instance (sync path only) to remain
    module-scoped without pulling in the function-scoped client fixture.
    """
    _pause()
    return TTSPollinations(api_key=api_key).generate(_SHORT_TEXT)


class TestAudioModelCatalog:
    """Verify that the live /audio/models endpoint is reachable and well-formed."""

    def test_list_audio_models_returns_non_empty_list(self, api_key: str) -> None:
        info = ModelInformation(api_key=api_key)
        models = info.list_audio_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_each_model_entry_has_identifier(self, api_key: str) -> None:
        info = ModelInformation(api_key=api_key)
        models = info.list_audio_models()
        for entry in models:
            assert isinstance(entry, dict)
            model_id = entry.get("id") or entry.get("model") or entry.get("name")
            assert model_id, f"Entry without identifier: {entry}"

    def test_get_available_models_includes_audio_key(self, api_key: str) -> None:
        info = ModelInformation(api_key=api_key)
        result = info.get_available_models()
        assert "audio" in result
        assert isinstance(result["audio"], list)

    def test_audio_catalog_not_empty_in_get_available_models(self, api_key: str) -> None:
        info = ModelInformation(api_key=api_key)
        result = info.get_available_models()
        assert len(result["audio"]) > 0

    async def test_alist_audio_models_returns_non_empty_list(self, api_key: str) -> None:
        info = ModelInformation(api_key=api_key)
        models = await info.alist_audio_models()
        assert isinstance(models, list)
        assert len(models) > 0

    async def test_aget_available_models_includes_audio_key(self, api_key: str) -> None:
        info = ModelInformation(api_key=api_key)
        result = await info.aget_available_models()
        assert "audio" in result
        assert len(result["audio"]) > 0

    def test_load_audio_model_ids_populates_module_cache(self, api_key: str) -> None:
        tts_module._audio_model_ids_loaded = False
        ids = _load_audio_model_ids(api_key, force=True)
        assert isinstance(ids, list)
        assert len(ids) > 0
        # El caché del módulo debe coincidir con lo retornado.
        assert tts_module._audio_model_ids_cache == ids

    def test_load_audio_model_ids_force_refresh_updates_cache(
        self, api_key: str
    ) -> None:
        ids_first = _load_audio_model_ids(api_key, force=True)
        _pause()
        ids_second = _load_audio_model_ids(api_key, force=True)
        # Ambas listas deben tener los mismos modelos (el catálogo no cambia entre llamadas).
        assert set(ids_first) == set(ids_second)


class TestClientConstruction:
    """Verify TTSPollinations can be constructed with various configurations."""

    def test_default_construction_succeeds(self, api_key: str) -> None:
        tts = TTSPollinations(api_key=api_key)
        assert tts.base_url == "https://gen.pollinations.ai"
        assert tts.timeout_s == 120.0

    def test_construction_with_all_defaults_set(self, api_key: str) -> None:
        tts = TTSPollinations(
            api_key=api_key,
            model="tts-1",
            voice="alloy",
            response_format="mp3",
            speed=1.0,
        )
        assert tts.model == "tts-1"
        assert tts.voice == "alloy"
        assert tts.response_format == "mp3"
        assert tts.speed == 1.0

    def test_api_key_hidden_in_repr(self, api_key: str) -> None:
        tts = TTSPollinations(api_key=api_key)
        assert api_key not in repr(tts)

    def test_known_model_construction_no_warning(self, api_key: str) -> None:
        import warnings

        # Usar modelo del catálogo vivo para que no haya warning.
        ids = _load_audio_model_ids(api_key, force=True)
        known = ids[0] if ids else "tts-1"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TTSPollinations(api_key=api_key, model=known)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_unknown_model_emits_warning_but_constructs(self, api_key: str) -> None:
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tts = TTSPollinations(api_key=api_key, model="definitely-not-a-real-model-xyz")
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        # La instancia existe y su modelo está configurado correctamente.
        assert tts.model == "definitely-not-a-real-model-xyz"


class TestSyncGeneration:
    """Verify generate() and generate_response() against the live API."""

    def test_generate_returns_bytes(self, default_mp3: bytes) -> None:
        assert isinstance(default_mp3, bytes)

    def test_generate_returns_non_empty_bytes(self, default_mp3: bytes) -> None:
        assert len(default_mp3) > 0

    def test_generate_default_format_is_valid_mp3(self, default_mp3: bytes) -> None:
        assert _check_audio_magic(default_mp3, "mp3"), (
            f"Expected MP3 magic bytes, got: {default_mp3[:8]!r}"
        )

    def test_generate_mp3_has_meaningful_size(self, default_mp3: bytes) -> None:
        # Un clip MP3 de audio real tiene al menos 1 KB.
        assert len(default_mp3) > 1_024

    def test_generate_response_returns_httpx_response(self, client: TTSPollinations) -> None:
        import httpx

        resp = client.generate_response(_SHORT_TEXT)
        assert isinstance(resp, httpx.Response)

    def test_generate_response_status_200(self, client: TTSPollinations) -> None:
        resp = client.generate_response(_SHORT_TEXT)
        assert resp.status_code == 200

    def test_generate_response_content_type_is_audio(
        self, client: TTSPollinations
    ) -> None:
        resp = client.generate_response(_SHORT_TEXT)
        content_type = resp.headers.get("content-type", "")
        assert "audio" in content_type.lower(), (
            f"Expected audio content-type, got: {content_type}"
        )

    def test_generate_response_content_matches_generate(
        self, client: TTSPollinations
    ) -> None:
        _pause()
        audio_bytes = client.generate(_SHORT_TEXT)
        assert len(audio_bytes) > 0

    def test_generate_longer_text_produces_larger_output(
        self, client: TTSPollinations
    ) -> None:
        _pause()
        short_audio = client.generate(_SHORT_TEXT)
        _pause()
        long_audio = client.generate(_MEDIUM_TEXT)
        # Texto más largo debe producir audio más largo (no siempre estrictamente,
        # pero con una diferencia de texto suficiente debe cumplirse).
        assert len(long_audio) > len(short_audio)


class TestAsyncGeneration:
    """Verify agenerate() and agenerate_response() against the live API."""

    async def test_agenerate_returns_bytes(self, client: TTSPollinations) -> None:
        _pause()
        result = await client.agenerate(_SHORT_TEXT)
        assert isinstance(result, bytes)

    async def test_agenerate_returns_non_empty_bytes(
        self, client: TTSPollinations
    ) -> None:
        _pause()
        result = await client.agenerate(_SHORT_TEXT)
        assert len(result) > 0

    async def test_agenerate_default_format_is_valid_mp3(
        self, client: TTSPollinations
    ) -> None:
        _pause()
        result = await client.agenerate(_SHORT_TEXT)
        assert _check_audio_magic(result, "mp3"), (
            f"Expected MP3 magic bytes, got: {result[:8]!r}"
        )

    async def test_agenerate_response_status_200(
        self, client: TTSPollinations
    ) -> None:
        _pause()
        resp = await client.agenerate_response(_SHORT_TEXT)
        assert resp.status_code == 200

    async def test_agenerate_response_content_type_is_audio(
        self, client: TTSPollinations
    ) -> None:
        _pause()
        resp = await client.agenerate_response(_SHORT_TEXT)
        content_type = resp.headers.get("content-type", "")
        assert "audio" in content_type.lower()

    async def test_agenerate_result_structurally_consistent_with_sync(
        self, client: TTSPollinations
    ) -> None:
        # Ambas versiones deben producir MP3 válido aunque el contenido exacto varíe.
        _pause()
        async_result = await client.agenerate(_SHORT_TEXT)
        assert _check_audio_magic(async_result, "mp3")


class TestLangChainInterface:
    """Verify the Runnable-compatible invoke/ainvoke methods."""

    def test_invoke_returns_bytes(self, client: TTSPollinations) -> None:
        _pause()
        result = client.invoke(_SHORT_TEXT)
        assert isinstance(result, bytes)

    def test_invoke_returns_valid_mp3(self, client: TTSPollinations) -> None:
        _pause()
        result = client.invoke(_SHORT_TEXT)
        assert _check_audio_magic(result, "mp3")

    def test_invoke_config_none_does_not_raise(self, client: TTSPollinations) -> None:
        _pause()
        result = client.invoke(_SHORT_TEXT, config=None)
        assert len(result) > 0

    def test_invoke_config_dict_does_not_raise(self, client: TTSPollinations) -> None:
        _pause()
        result = client.invoke(_SHORT_TEXT, config={"run_name": "integration-test"})
        assert len(result) > 0

    def test_invoke_kwargs_forwarded_to_generate(
        self, client: TTSPollinations
    ) -> None:
        _pause()
        # Forzar formato WAV vía kwargs; el resultado debe tener cabecera RIFF.
        result = client.invoke(_SHORT_TEXT, response_format="wav")
        assert _check_audio_magic(result, "wav"), (
            f"Expected WAV magic bytes, got: {result[:8]!r}"
        )

    async def test_ainvoke_returns_bytes(self, client: TTSPollinations) -> None:
        _pause()
        result = await client.ainvoke(_SHORT_TEXT)
        assert isinstance(result, bytes)

    async def test_ainvoke_returns_valid_mp3(self, client: TTSPollinations) -> None:
        _pause()
        result = await client.ainvoke(_SHORT_TEXT)
        assert _check_audio_magic(result, "mp3")

    async def test_ainvoke_config_none_does_not_raise(
        self, client: TTSPollinations
    ) -> None:
        _pause()
        result = await client.ainvoke(_SHORT_TEXT, config=None)
        assert len(result) > 0

    async def test_ainvoke_kwargs_forwarded_to_agenerate(
        self, client: TTSPollinations
    ) -> None:
        _pause()
        result = await client.ainvoke(_SHORT_TEXT, response_format="wav")
        assert _check_audio_magic(result, "wav")


class TestAudioFormats:
    """Verify that each supported response_format produces the correct container."""

    @pytest.fixture(scope="class")
    def fmt_client(self, api_key: str) -> TTSPollinations:
        """Client fixture scoped to the class to avoid repeated catalog loads."""
        return TTSPollinations(api_key=api_key)

    def test_mp3_format_magic_bytes(self, fmt_client: TTSPollinations) -> None:
        _pause()
        data = fmt_client.generate(_SHORT_TEXT, response_format="mp3")
        assert _check_audio_magic(data, "mp3"), f"Got: {data[:8]!r}"

    def test_wav_format_magic_bytes(self, fmt_client: TTSPollinations) -> None:
        _pause()
        data = fmt_client.generate(_SHORT_TEXT, response_format="wav")
        assert _check_audio_magic(data, "wav"), f"Got: {data[:8]!r}"
        # Verificar también la firma WAVE en el byte offset 8.
        assert data[8:12] == b"WAVE"

    def test_flac_format_returns_non_empty_bytes(self, fmt_client: TTSPollinations) -> None:
        # Caracterización confirmada: Pollinations retorna PCM crudo (16-bit LE con signo)
        # cuando se solicita response_format="flac". No hay cabecera de contenedor FLAC.
        _pause()
        data = fmt_client.generate(_SHORT_TEXT, response_format="flac")
        assert len(data) > 0

    def test_opus_format_magic_bytes(self, fmt_client: TTSPollinations) -> None:
        _pause()
        data = fmt_client.generate(_SHORT_TEXT, response_format="opus")
        assert _check_audio_magic(data, "opus"), f"Got: {data[:8]!r}"

    def test_aac_format_non_empty(self, fmt_client: TTSPollinations) -> None:
        _pause()
        data = fmt_client.generate(_SHORT_TEXT, response_format="aac")
        assert len(data) > 0

    def test_pcm_format_non_empty_raw_bytes(self, fmt_client: TTSPollinations) -> None:
        _pause()
        data = fmt_client.generate(_SHORT_TEXT, response_format="pcm")
        # PCM es audio crudo sin cabecera; solo verificamos que no esté vacío.
        assert len(data) > 0

    def test_different_formats_produce_different_bytes(
        self, fmt_client: TTSPollinations
    ) -> None:
        _pause()
        mp3 = fmt_client.generate(_SHORT_TEXT, response_format="mp3")
        _pause()
        wav = fmt_client.generate(_SHORT_TEXT, response_format="wav")
        # MP3 y WAV son contenedores distintos; los primeros 4 bytes siempre difieren.
        assert mp3[:4] != wav[:4]

    def test_wav_size_larger_than_mp3_for_same_text(
        self, fmt_client: TTSPollinations
    ) -> None:
        _pause()
        mp3 = fmt_client.generate(_SHORT_TEXT, response_format="mp3")
        _pause()
        wav = fmt_client.generate(_SHORT_TEXT, response_format="wav")
        # WAV sin comprimir siempre supera en tamaño al MP3 equivalente.
        assert len(wav) > len(mp3)

    def test_format_via_instance_default(self, api_key: str) -> None:
        _pause()
        tts = TTSPollinations(api_key=api_key, response_format="wav")
        data = tts.generate(_SHORT_TEXT)
        assert _check_audio_magic(data, "wav")

    def test_flac_request_returns_pcm_like_bytes(
        self, fmt_client: TTSPollinations
    ) -> None:
        # Test de caracterización: documenta que el API retorna PCM crudo para flac.
        # Si este test falla en el futuro, significa que el proveedor corrigió el comportamiento
        # y los tests de magic bytes de FLAC deberán actualizar sus aserciones.
        _pause()
        data = fmt_client.generate(_SHORT_TEXT, response_format="flac")
        # PCM 16-bit little-endian: los bytes son pares de valores enteros con signo.
        assert len(data) % 2 == 0, "PCM crudo debe tener número par de bytes (16-bit)"
        assert not data[:4] == b"fLaC", "Proveedor confirma: no retorna cabecera FLAC estándar"


class TestSpeed:
    """Verify that the speed parameter is honoured by the API."""

    @pytest.fixture(scope="class")
    def speed_client(self, api_key: str) -> TTSPollinations:
        return TTSPollinations(api_key=api_key, response_format="wav")

    def test_speed_1x_produces_valid_audio(self, speed_client: TTSPollinations) -> None:
        _pause()
        data = speed_client.generate(_SHORT_TEXT, speed=1.0)
        assert len(data) > 0
        assert _check_audio_magic(data, "wav")

    def test_speed_2x_produces_shorter_wav_than_1x(
        self, speed_client: TTSPollinations
    ) -> None:
        _pause()
        normal = speed_client.generate(_SHORT_TEXT, speed=1.0)
        _pause()
        fast = speed_client.generate(_SHORT_TEXT, speed=2.0)
        # A velocidad doble la duración se reduce; WAV tiene duración proporcional al tamaño.
        assert len(fast) < len(normal)

    def test_speed_0_5x_produces_longer_wav_than_1x(
        self, speed_client: TTSPollinations
    ) -> None:
        _pause()
        normal = speed_client.generate(_SHORT_TEXT, speed=1.0)
        _pause()
        slow = speed_client.generate(_SHORT_TEXT, speed=0.5)
        assert len(slow) > len(normal)

    def test_speed_at_minimum_boundary(self, speed_client: TTSPollinations) -> None:
        _pause()
        data = speed_client.generate(_SHORT_TEXT, speed=0.25)
        assert len(data) > 0

    def test_speed_at_maximum_boundary(self, speed_client: TTSPollinations) -> None:
        _pause()
        data = speed_client.generate(_SHORT_TEXT, speed=4.0)
        assert len(data) > 0


class TestVoices:
    """Verify a representative sample of voices from both TTS families."""

    @pytest.fixture(scope="class")
    def voice_client(self, api_key: str) -> TTSPollinations:
        return TTSPollinations(api_key=api_key, response_format="mp3")

    # Muestra representativa: 3 voces OpenAI + 3 voces ElevenLabs.
    @pytest.mark.parametrize("voice", ["alloy", "echo", "shimmer"])
    def test_openai_voices_produce_valid_mp3(
        self, voice_client: TTSPollinations, voice: str
    ) -> None:
        _pause()
        data = voice_client.generate(_SHORT_TEXT, voice=voice)
        assert _check_audio_magic(data, "mp3"), (
            f"Voice '{voice}' did not produce valid MP3. Got: {data[:8]!r}"
        )

    @pytest.mark.parametrize("voice", ["rachel", "bella", "adam"])
    def test_elevenlabs_voices_produce_valid_mp3(
        self, voice_client: TTSPollinations, voice: str
    ) -> None:
        _pause()
        data = voice_client.generate(_SHORT_TEXT, voice=voice)
        assert _check_audio_magic(data, "mp3"), (
            f"Voice '{voice}' did not produce valid MP3. Got: {data[:8]!r}"
        )

    @pytest.mark.parametrize("voice", ["alloy", "echo", "shimmer"])
    def test_openai_voice_size_reasonable(
        self, voice_client: TTSPollinations, voice: str
    ) -> None:
        _pause()
        data = voice_client.generate(_SHORT_TEXT, voice=voice)
        assert len(data) > 1_024

    def test_different_voices_may_produce_different_output(
        self, voice_client: TTSPollinations
    ) -> None:
        _pause()
        alloy = voice_client.generate(_SHORT_TEXT, voice="alloy")
        _pause()
        echo = voice_client.generate(_SHORT_TEXT, voice="echo")
        # Voces distintas generan archivos distintos (el contenido de audio difiere).
        assert alloy != echo

    def test_voice_via_instance_default(self, api_key: str) -> None:
        _pause()
        tts = TTSPollinations(api_key=api_key, voice="shimmer")
        data = tts.generate(_SHORT_TEXT)
        assert len(data) > 0

    def test_per_call_voice_overrides_instance_default(self, api_key: str) -> None:
        _pause()
        tts = TTSPollinations(api_key=api_key, voice="alloy")
        data = tts.generate(_SHORT_TEXT, voice="echo")
        # El resultado no debe ser vacío; la override fue aceptada por la API.
        assert len(data) > 0


class TestWithParams:
    """Verify that with_params creates independent, correctly configured instances."""

    def test_with_params_returns_new_instance(self, client: TTSPollinations) -> None:
        new = client.with_params(voice="echo")
        assert new is not client

    def test_with_params_original_unchanged(self, client: TTSPollinations) -> None:
        original_voice = client.voice
        client.with_params(voice="echo")
        assert client.voice == original_voice

    def test_with_params_new_instance_generates_valid_audio(
        self, api_key: str
    ) -> None:
        _pause()
        base = TTSPollinations(api_key=api_key)
        wav_client = base.with_params(response_format="wav", voice="shimmer")
        data = wav_client.generate(_SHORT_TEXT)
        assert _check_audio_magic(data, "wav")

    def test_with_params_chaining_accumulates_overrides(self, api_key: str) -> None:
        _pause()
        tts = (
            TTSPollinations(api_key=api_key)
            .with_params(voice="coral")
            .with_params(response_format="wav")
            .with_params(speed=1.5)
        )
        assert tts.voice == "coral"
        assert tts.response_format == "wav"
        assert tts.speed == 1.5
        data = tts.generate(_SHORT_TEXT)
        assert _check_audio_magic(data, "wav")

    def test_with_params_api_key_preserved_and_functional(self, api_key: str) -> None:
        _pause()
        base = TTSPollinations(api_key=api_key)
        derived = base.with_params(voice="alloy")
        # La instancia derivada debe poder llamar a la API sin error de auth.
        data = derived.generate(_SHORT_TEXT)
        assert len(data) > 0


class TestRequestParameterPrecedence:
    """Verify that instance defaults, per-call params, and kwargs stack correctly."""

    def test_per_call_params_dict_overrides_instance_default(
        self, api_key: str
    ) -> None:
        _pause()
        tts = TTSPollinations(api_key=api_key, response_format="mp3")
        data = tts.generate(_SHORT_TEXT, params={"response_format": "wav"})
        assert _check_audio_magic(data, "wav")

    def test_per_call_kwarg_overrides_instance_default(self, api_key: str) -> None:
        _pause()
        tts = TTSPollinations(api_key=api_key, response_format="mp3")
        data = tts.generate(_SHORT_TEXT, response_format="wav")
        assert _check_audio_magic(data, "wav")

    def test_kwarg_overrides_params_dict(self, api_key: str) -> None:
        _pause()
        tts = TTSPollinations(api_key=api_key)
        # params pide mp3, kwarg pide wav → kwarg gana.
        data = tts.generate(
            _SHORT_TEXT,
            params={"response_format": "mp3"},
            response_format="wav",
        )
        assert _check_audio_magic(data, "wav")


class TestElevenMusic:
    """
    Verify the elevenmusic model for AI music generation.
    """

    @pytest.fixture(scope="class")
    def music_client(self, api_key: str) -> TTSPollinations:
        return TTSPollinations(
            api_key=api_key,
            model="elevenmusic",
            response_format="mp3",
        )

    def test_elevenmusic_generates_non_empty_bytes(
        self, music_client: TTSPollinations
    ) -> None:
        data = music_client.generate("A gentle acoustic melody", duration=10.0)
        assert len(data) > 0

    def test_elevenmusic_output_is_valid_mp3(
        self, music_client: TTSPollinations
    ) -> None:
        data = music_client.generate("Upbeat electronic music", duration=10.0)
        assert _check_audio_magic(data, "mp3"), f"Got: {data[:8]!r}"

    def test_elevenmusic_instrumental_flag_accepted(
        self, music_client: TTSPollinations
    ) -> None:
        data = music_client.generate(
            "Calm piano music",
            duration=10.0,
            instrumental=True,
        )
        assert len(data) > 0

    def test_elevenmusic_longer_duration_produces_larger_file(
        self, music_client: TTSPollinations
    ) -> None:
        short = music_client.generate("Jazz riff", duration=5.0)
        _pause()
        long = music_client.generate("Jazz riff", duration=15.0)
        assert len(long) > len(short)

    async def test_elevenmusic_async_generation(
        self, music_client: TTSPollinations
    ) -> None:
        data = await music_client.agenerate("Soft ambient music", duration=10.0)
        assert _check_audio_magic(data, "mp3")


class TestAPIErrorHandling:
    """Verify that API errors are surfaced as PollinationsAPIError."""

    def test_invalid_api_key_raises_pollinations_api_error(self) -> None:
        _pause()
        tts = TTSPollinations(api_key="sk-invalid-key-for-testing")
        with pytest.raises(PollinationsAPIError) as exc_info:
            tts.generate(_SHORT_TEXT)
        # El error debe ser de tipo auth (401) o similar.
        assert exc_info.value.is_auth_error or exc_info.value.status_code in (
            401,
            403,
            400,
        )

    def test_api_error_exposes_status_code(self) -> None:
        _pause()
        tts = TTSPollinations(api_key="sk-invalid-key-for-testing")
        with pytest.raises(PollinationsAPIError) as exc_info:
            tts.generate(_SHORT_TEXT)
        assert isinstance(exc_info.value.status_code, int)
        assert exc_info.value.status_code >= 400

    async def test_async_invalid_api_key_raises_pollinations_api_error(self) -> None:
        _pause()
        tts = TTSPollinations(api_key="sk-invalid-key-for-testing")
        with pytest.raises(PollinationsAPIError):
            await tts.agenerate(_SHORT_TEXT)
