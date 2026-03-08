"""
TTS (Text-to-Speech) client for the Pollinations AI API.

Provides `TTSPollinations`, a configurable wrapper for the
``POST /v1/audio/speech`` endpoint, with dynamic audio model catalog
loading, full ``CreateSpeechRequest`` schema support, binary audio
response handling, and a LangChain Runnable-compatible interface.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from langchain_core.runnables import Runnable

from langchain_pollinations._audio_catalog import (
    _audio_model_ids_cache,
    _load_audio_model_ids,
    _audio_model_ids_loaded,  # noqa: F401
)
from langchain_pollinations._auth import AuthConfig
from langchain_pollinations._client import HttpConfig, PollinationsHttpClient

DEFAULT_BASE_URL = "https://gen.pollinations.ai"

# Formato de audio de salida; enum cerrado según la spec de la API.
AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

# Tipo de voz. Se usa str para tolerar adiciones futuras al catálogo del API
# sin provocar ValidationError. Voces conocidas al momento del release (34):
#   alloy, echo, fable, onyx, shimmer, ash, ballad, coral, sage, verse,
#   rachel, domi, bella, elli, charlotte, dorothy, sarah, emily, lily,
#   matilda, adam, antoni, arnold, josh, sam, daniel, charlie, james,
#   fin, callum, liam, george, brian, bill
VoiceId = str


class SpeechRequest(BaseModel):
    """
    Typed representation of the ``CreateSpeechRequest`` schema used by
    ``POST /v1/audio/speech``.

    All field names and defaults mirror the OpenAPI spec exactly. The
    ``to_body()`` method serialises the model to a dict suitable for use
    as a JSON POST body, omitting ``None`` fields so that model-specific
    optional parameters (``duration``, ``instrumental``) are not transmitted
    for models that do not support them.

    Attributes:
        input: Text to synthesise. Required. 1–4096 characters.
        model: Audio model identifier. ``None`` defers to the API default.
        voice: Voice to use. Defaults to ``"alloy"``.
        response_format: Output audio encoding. Defaults to ``"mp3"``.
        speed: Playback speed multiplier, 0.25–4.0. Defaults to ``1.0``.
        duration: Music duration in seconds, 3–300. ``elevenmusic`` only.
            ``None`` omits the field from the request body.
        instrumental: When ``True``, guarantees no vocals in the output.
            ``elevenmusic`` only. ``None`` omits the field from the request body.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Único campo required según la spec.
    input: str = Field(..., min_length=1, max_length=4096)

    # Campos opcionales con defaults del API; None → omitido del body.
    model: Optional[str] = None
    voice: VoiceId = "alloy"
    response_format: AudioFormat = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    # Parámetros exclusivos de elevenmusic; None → omitido del body con exclude_none=True.
    duration: Optional[float] = Field(default=None, ge=3.0, le=300.0)
    instrumental: Optional[bool] = None

    def to_body(self) -> dict[str, Any]:
        """
        Serialise the request to a JSON-ready dictionary.

        ``None`` fields are excluded so that model-specific parameters are
        not transmitted when they are not applicable to the chosen model.

        Returns:
            A dictionary suitable for use as a POST JSON body.
        """
        return self.model_dump(exclude_none=True)


class TTSPollinations(BaseModel, Runnable[str, bytes]):
    """
    Configurable client for the Pollinations Text-to-Speech endpoint.

    Wraps ``POST /v1/audio/speech`` with dynamic audio model catalog
    validation, instance-level parameter defaults, and a LangChain
    Runnable-compatible interface (``invoke`` / ``ainvoke``).

    All configuration fields are optional. When ``None``, a field is
    excluded from the request body and the API applies its own defaults.
    Fields set at instance level serve as per-instance defaults that can be
    overridden on a per-call basis via `generate` / `agenerate`
    or replaced in bulk via `with_params`.

    Example — basic usage::

        tts = TTSPollinations(voice="rachel", response_format="mp3")
        audio_bytes = tts.generate("Hello, world!")

    Example — elevenmusic::

        music = TTSPollinations(model="elevenmusic", duration=30, instrumental=True)
        audio_bytes = music.generate("An upbeat jazz theme")

    Example — LangChain chain::

        from langchain_core.runnables import RunnableLambda
        pipeline = RunnableLambda(lambda x: x["text"]) | tts
        audio_bytes = pipeline.invoke({"text": "Hello"})

    Attributes:
        api_key: Pollinations API key. Falls back to ``POLLINATIONS_API_KEY``
            environment variable when ``None``.
        base_url: API base URL. Defaults to ``https://gen.pollinations.ai``.
        timeout_s: HTTP request timeout in seconds. Defaults to ``120.0``.
        model: Default audio model. ``None`` defers to the API default.
        voice: Default voice identifier. ``None`` defers to the API default
            (``"alloy"``).
        response_format: Default output audio format. ``None`` defers to the
            API default (``"mp3"``).
        speed: Default playback speed, 0.25–4.0. ``None`` defers to ``1.0``.
        duration: Default music duration in seconds, 3–300.
            ``elevenmusic`` only.
        instrumental: Default instrumental flag. ``elevenmusic`` only.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", populate_by_name=True)

    # Requerido por Runnable: identificador opcional del runnable para logs y trazas LC.
    name: Optional[str] = None

    # --- Auth / transporte ---
    api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 120.0

    # --- Defaults del request ---
    model: Optional[str] = None
    voice: Optional[VoiceId] = None
    response_format: Optional[AudioFormat] = None
    speed: Optional[float] = Field(default=None, ge=0.25, le=4.0)

    # Parámetros exclusivos de elevenmusic.
    duration: Optional[float] = Field(default=None, ge=3.0, le=300.0)
    instrumental: Optional[bool] = None

    _http: PollinationsHttpClient = PrivateAttr()

    @field_validator("model", mode="after")
    @classmethod
    def _validate_model_id(cls, v: str | None) -> str | None:
        """
        Emit a ``UserWarning`` when the provided model ID is absent from the
        known audio model catalog, without blocking the request.

        Validation is intentionally non-strict so that models newly added to
        the API continue to work even before the local catalog is refreshed.
        Only non-``None`` values are checked; ``None`` means "use the API
        default" and is always accepted without warning.

        Args:
            v: The model ID string to validate, or ``None`` when unset.

        Returns:
            The original value unchanged.
        """
        if v is not None and _audio_model_ids_cache and v not in _audio_model_ids_cache:
            warnings.warn(
                f"Audio model ID '{v}' is not in the known audio model catalog "
                f"({len(_audio_model_ids_cache)} models loaded). "
                "The request will proceed, but the API may return an error if the ID is invalid. "
                "Call _load_audio_model_ids(force=True) to refresh the catalog from the API.",
                UserWarning,
                stacklevel=2,
            )
        return v

    def __init__(self, **data: Any) -> None:
        """
        Initialise the TTS client, loading the audio model catalog before
        Pydantic field validation runs.

        The catalog load is a no-op when the catalog was already fetched in
        this process (one-shot pattern). Call
        ``_load_audio_model_ids(force=True)`` explicitly to force a refresh.

        Args:
            **data: Configuration parameters forwarded to Pydantic.
        """
        # Cargar el catálogo ANTES de que super().__init__ ejecute los field_validators
        # (incluyendo _validate_model_id). Se extrae api_key de data porque self aún no existe.
        _load_audio_model_ids(data.get("api_key"))
        super().__init__(**data)
        auth = AuthConfig.from_env_or_value(self.api_key)
        self._http = PollinationsHttpClient(
            config=HttpConfig(base_url=self.base_url, timeout_s=self.timeout_s),
            api_key=auth.api_key,
        )

    def with_params(self, **overrides: Any) -> TTSPollinations:
        """
        Create a new instance with updated default parameters.

        The current instance is not modified. The ``api_key`` is explicitly
        preserved in the copy because it is excluded from ``model_dump``.

        Args:
            **overrides: Parameter values to override in the new instance.

        Returns:
            A new `TTSPollinations` instance with merged configuration.
        """
        merged = self.model_dump(exclude_none=False)
        merged.update(overrides)
        # api_key se excluye del dump por `exclude=True`; reinyectar manualmente.
        merged["api_key"] = self.api_key
        return TTSPollinations(**merged)

    def _defaults_dict(self) -> dict[str, Any]:
        """
        Extract the current instance configuration into a clean dictionary,
        excluding all ``None`` values.

        Returns:
            A dictionary containing only the parameters explicitly configured
            on this instance, ready to serve as the base of a request body.
        """
        out: dict[str, Any] = {}
        for k in ("model", "voice", "response_format", "speed", "duration", "instrumental"):
            v = getattr(self, k, None)
            if v is not None:
                out[k] = v
        return out

    def _build_body(
        self, text: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Build and validate the final JSON body for a TTS API request.

        Merges instance defaults, per-call ``params``, and ``kwargs`` in
        ascending priority order, then injects the required ``input`` field
        and validates the full payload through `SpeechRequest`.

        Args:
            text: The text to synthesise. Mapped to the ``input`` field.
                Always takes final precedence over any other source.
            params: Optional dictionary of per-call parameter overrides.
            **kwargs: Additional per-call overrides as keyword arguments.

        Returns:
            A validated dictionary suitable for use as a POST JSON body.

        Raises:
            pydantic.ValidationError: If the merged parameters fail
                `SpeechRequest` validation (e.g. ``speed`` out of range).
        """
        merged: dict[str, Any] = {}
        # Defaults de instancia como base de la mezcla.
        merged.update(self._defaults_dict())
        # Los overrides por llamada tienen mayor prioridad que los defaults de instancia.
        if params:
            merged.update(dict(params))
        if kwargs:
            merged.update(kwargs)
        # El texto siempre proviene del argumento posicional y tiene la máxima prioridad.
        merged["input"] = text
        validated = SpeechRequest(**merged)
        return validated.to_body()

    def generate_response(
        self, text: str, *, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> httpx.Response:
        """
        Execute a synchronous POST request to the TTS endpoint and return the
        raw HTTP response.

        Use `generate` to receive the audio bytes directly.

        Args:
            text: The text to synthesise.
            params: Optional per-call parameter overrides.
            **kwargs: Additional per-call overrides as keyword arguments.

        Returns:
            The raw `httpx.Response` from the API. The response body
            contains binary audio in the format requested via ``response_format``.

        Raises:
            PollinationsAPIError: On any non-2xx HTTP response.
        """
        body = self._build_body(text, params, **kwargs)
        return self._http.post_json("/v1/audio/speech", body)

    async def agenerate_response(
        self, text: str, *, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> httpx.Response:
        """
        Execute an asynchronous POST request to the TTS endpoint and return
        the raw HTTP response.

        Use `agenerate` to receive the audio bytes directly.

        Args:
            text: The text to synthesise.
            params: Optional per-call parameter overrides.
            **kwargs: Additional per-call overrides as keyword arguments.

        Returns:
            The raw `httpx.Response` from the API. The response body
            contains binary audio in the format requested via ``response_format``.

        Raises:
            PollinationsAPIError: On any non-2xx HTTP response.
        """
        body = self._build_body(text, params, **kwargs)
        return await self._http.apost_json("/v1/audio/speech", body)

    def generate(
        self, text: str, *, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> bytes:
        """
        Generate audio synchronously and return the raw audio bytes.

        Args:
            text: The text to synthesise.
            params: Optional per-call parameter overrides.
            **kwargs: Additional per-call overrides as keyword arguments.

        Returns:
            The binary audio content produced by the API, in the format
            specified by ``response_format`` (default ``"mp3"``).

        Raises:
            PollinationsAPIError: On any non-2xx HTTP response.
        """
        return self.generate_response(text, params=params, **kwargs).content

    async def agenerate(
        self, text: str, *, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> bytes:
        """
        Generate audio asynchronously and return the raw audio bytes.

        Args:
            text: The text to synthesise.
            params: Optional per-call parameter overrides.
            **kwargs: Additional per-call overrides as keyword arguments.

        Returns:
            The binary audio content produced by the API, in the format
            specified by ``response_format`` (default ``"mp3"``).

        Raises:
            PollinationsAPIError: On any non-2xx HTTP response.
        """
        r = await self.agenerate_response(text, params=params, **kwargs)
        return r.content

    def invoke(self, input: str, config: Any | None = None, **kwargs: Any) -> bytes:
        """
        LangChain-compatible synchronous invocation method.

        Implements the Runnable duck-typing interface expected by LangChain
        v1.2.x+. The ``config`` parameter is accepted for interface
        compatibility but is not used internally.

        Args:
            input: The text to synthesise.
            config: Optional LangChain ``RunnableConfig`` (unused).
            **kwargs: Additional generation parameters forwarded to
                `generate`.

        Returns:
            The binary audio content produced by the API.
        """
        _ = config
        return self.generate(input, **kwargs)

    async def ainvoke(self, input: str, config: Any | None = None, **kwargs: Any) -> bytes:
        """
        LangChain-compatible asynchronous invocation method.

        Implements the async Runnable duck-typing interface expected by
        LangChain v1.2.x+. The ``config`` parameter is accepted for interface
        compatibility but is not used internally.

        Args:
            input: The text to synthesise.
            config: Optional LangChain ``RunnableConfig`` (unused).
            **kwargs: Additional generation parameters forwarded to
                `agenerate`.

        Returns:
            The binary audio content produced by the API.
        """
        _ = config
        return await self.agenerate(input, **kwargs)
