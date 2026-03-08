"""
STT (Speech-to-Text) client for the Pollinations AI API.

Provides `STTPollinations`, a configurable wrapper for the
``POST /v1/audio/transcriptions`` endpoint, with dynamic audio model catalog
validation, full multipart/form-data request construction, typed response
parsing, and a LangChain Runnable-compatible interface.
"""

from __future__ import annotations

import os
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

# Formatos de salida de la transcripción; enum cerrado según la spec.
TranscriptionFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]

# Formatos de audio aceptados como entrada; enum cerrado según la spec.
AudioInputFormat = Literal["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]

# Mapeo de extensión (sin punto, minúsculas) → MIME type para la parte file
# del cuerpo multipart. httpx incluye el MIME type en el Content-Disposition.
_AUDIO_MIME_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "mp4": "audio/mp4",
    "mpeg": "audio/mpeg",
    "mpga": "audio/mpeg",
    "m4a": "audio/mp4",
    "wav": "audio/wav",
    "webm": "audio/webm",
}

# MIME type de fallback cuando la extensión no está en el mapeo.
_FALLBACK_MIME_TYPE = "application/octet-stream"


class TranscriptionParams(BaseModel):
    """
    Typed representation of the non-file form fields for
    ``POST /v1/audio/transcriptions``.

    The audio file itself is handled separately so that `to_form_data`
    can produce a clean string dict suitable for the ``data=`` parameter of
    an httpx multipart POST.

    Attributes:
        model: STT model identifier. Defaults to ``"whisper-large-v3"``.
        language: ISO-639-1 language code (e.g. ``"en"``, ``"es"``).
            Improves transcription accuracy. ``None`` enables automatic
            language detection.
        prompt: Optional context text to guide transcription style or to
            continue a previous audio segment. ``None`` omits the field.
        response_format: Output format for the transcription. Defaults to
            ``"json"``.
        temperature: Sampling temperature in the range 0.0–1.0. Lower values
            produce more deterministic output. ``None`` defers to the model
            default and omits the field from the request.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model: str = "whisper-large-v3"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: TranscriptionFormat = "json"
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def to_form_data(self) -> dict[str, str]:
        """
        Serialise the parameters to a string dictionary suitable for the
        ``data=`` parameter of an httpx multipart POST.

        ``None`` fields are excluded. All values are explicitly cast to
        ``str`` because httpx requires form field values to be strings.

        Returns:
            A dictionary of form field names to string values.
        """
        raw = self.model_dump(exclude_none=True)
        # httpx requiere que todos los valores en data= sean strings.
        return {k: str(v) for k, v in raw.items()}


class TranscriptionResponse(BaseModel):
    """
    Typed representation of the JSON response body from
    ``POST /v1/audio/transcriptions``.

    The ``text`` field contains the full transcription and is present for
    both ``"json"`` and ``"verbose_json"`` response formats. When
    ``"verbose_json"`` is requested, additional fields (``language``,
    ``duration``, ``segments``, ``words``, etc.) returned by the Whisper
    model are preserved via ``extra="allow"`` and accessible through
    ``instance.model_extra``.

    Attributes:
        text: The complete transcribed text produced by the model.
    """

    # extra="allow" para absorber los campos adicionales de verbose_json sin
    # romper la validación ante cambios del API.
    model_config = ConfigDict(extra="allow")

    text: str


class STTPollinations(BaseModel, Runnable[str, bytes]):
    """
    Configurable client for the Pollinations Speech-to-Text endpoint.

    Wraps ``POST /v1/audio/transcriptions`` with dynamic audio model catalog
    validation, instance-level parameter defaults, multipart/form-data request
    construction, typed response parsing, and a LangChain Runnable-compatible
    interface (``invoke`` / ``ainvoke``).

    All configuration fields are optional. When ``None``, a field is excluded
    from the request form and the API applies its own defaults. Fields set at
    instance level serve as per-instance defaults that can be overridden on a
    per-call basis via `transcribe` / `atranscribe` or replaced in
    bulk via `with_params`.

    Example — basic usage::

        stt = STTPollinations()
        with open("speech.mp3", "rb") as fh:
            result = stt.transcribe(fh.read())
        print(result.text)

    Example — with language hint and custom model::

        stt = STTPollinations(model="scribe", language="es")
        result = stt.transcribe(audio_bytes, file_name="grabacion.wav")
        print(result.text)

    Example — verbose JSON for segment-level data::

        stt = STTPollinations(response_format="verbose_json")
        result = stt.transcribe(audio_bytes)
        print(result.text)
        print(result.model_extra.get("segments"))

    Example — LangChain chain::

        pipeline = load_audio_stage | stt
        result = pipeline.invoke(audio_bytes)

    Attributes:
        api_key: Pollinations API key. Falls back to ``POLLINATIONS_API_KEY``
            environment variable when ``None``.
        base_url: API base URL. Defaults to ``https://gen.pollinations.ai``.
        timeout_s: HTTP request timeout in seconds. Defaults to ``120.0``.
        model: Default STT model. ``None`` defers to the API default
            (``"whisper-large-v3"``).
        language: Default ISO-639-1 language code. ``None`` enables automatic
            language detection.
        prompt: Default context prompt for transcription guidance.
        response_format: Default output format. ``None`` defers to ``"json"``.
        temperature: Default sampling temperature, 0.0–1.0.
        file_name: Default filename sent with the audio binary in the multipart
            body. The extension determines the MIME type. Defaults to
            ``"audio.mp3"``. Override per-call when the audio has a different
            format.
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
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[TranscriptionFormat] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Nombre de archivo de la parte file del multipart; la extensión determina
    # el MIME type. Puede solaparse por llamada vía transcribe(file_name=...).
    file_name: str = "audio.mp3"

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
        Initialise the STT client, loading the audio model catalog before
        Pydantic field validation runs.

        The catalog load is a no-op when the catalog was already fetched in
        this process (one-shot pattern). Call
        ``_load_audio_model_ids(force=True)`` explicitly to force a refresh.

        Args:
            **data: Configuration parameters forwarded to Pydantic.
        """
        # Cargar el catálogo ANTES de que super().__init__ ejecute los
        # field_validators (incluyendo _validate_model_id). Se extrae api_key
        # de data porque self aún no existe en este punto.
        _load_audio_model_ids(data.get("api_key"))
        super().__init__(**data)
        auth = AuthConfig.from_env_or_value(self.api_key)
        self._http = PollinationsHttpClient(
            config=HttpConfig(base_url=self.base_url, timeout_s=self.timeout_s),
            api_key=auth.api_key,
        )

    def with_params(self, **overrides: Any) -> STTPollinations:
        """
        Create a new instance with updated default parameters.

        The current instance is not modified. The ``api_key`` is explicitly
        preserved in the copy because it is excluded from ``model_dump``.

        Args:
            **overrides: Parameter values to override in the new instance.

        Returns:
            A new `STTPollinations` instance with merged configuration.
        """
        merged = self.model_dump(exclude_none=False)
        merged.update(overrides)
        # api_key se excluye del dump por `exclude=True`; reinyectar manualmente.
        merged["api_key"] = self.api_key
        return STTPollinations(**merged)

    def _defaults_dict(self) -> dict[str, Any]:
        """
        Extract the current instance configuration into a clean dictionary,
        excluding ``None`` values and the ``file_name`` field.

        ``file_name`` is intentionally excluded because it is not a form
        field recognised by ``TranscriptionParams``. It is extracted
        separately inside `_build_multipart`.

        Returns:
            A dictionary containing only the STT form parameters that are
            explicitly configured on this instance.
        """
        out: dict[str, Any] = {}
        for k in ("model", "language", "prompt", "response_format", "temperature"):
            v = getattr(self, k, None)
            if v is not None:
                out[k] = v
        return out

    @staticmethod
    def _mime_type_for(file_name: str) -> str:
        """
        Derive the audio MIME type from a filename's extension.

        Args:
            file_name: Filename whose extension will be inspected (e.g.
                ``"recording.wav"``).

        Returns:
            The corresponding MIME type string (e.g. ``"audio/wav"``), or
            ``"application/octet-stream"`` when the extension is not
            recognised.
        """
        _, ext = os.path.splitext(file_name)
        # Normalizar: quitar el punto y pasar a minúsculas antes de buscar.
        return _AUDIO_MIME_TYPES.get(ext.lstrip(".").lower(), _FALLBACK_MIME_TYPE)

    def _build_multipart(
        self,
        audio: bytes,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[
        dict[str, tuple[str, bytes, str]],
        dict[str, str],
        TranscriptionParams,
    ]:
        """
        Build the multipart/form-data components for a transcription request.

        Merges instance defaults, per-call ``params``, and ``kwargs`` in
        ascending priority order. The ``file_name`` key is extracted from the
        merged dict before constructing ``TranscriptionParams`` because it is
        not a recognised form field (``extra="forbid"`` would raise otherwise).

        Priority order (lowest → highest):
        1. Instance defaults from `_defaults_dict`.
        2. ``params`` dict (per-call overrides).
        3. ``**kwargs`` (per-call keyword overrides).

        Args:
            audio: Raw audio bytes to transcribe.
            params: Optional per-call parameter overrides. May include
                ``file_name`` to override the audio filename for this call.
            **kwargs: Additional per-call overrides. May include ``file_name``.

        Returns:
            A 3-tuple of:

            - ``files_dict``: ready for httpx ``files=`` parameter.
            - ``form_data``: ready for httpx ``data=`` parameter.
            - ``params_obj``: validated `TranscriptionParams` instance,
              used by callers to inspect ``response_format`` for response
              parsing without re-merging.

        Raises:
            pydantic.ValidationError: If the merged parameters fail
                `TranscriptionParams`` validation (e.g. ``temperature``
                out of range).
        """
        merged: dict[str, Any] = {}
        # Defaults de instancia como base de la mezcla.
        merged.update(self._defaults_dict())
        # Overrides por llamada tienen mayor prioridad que los defaults.
        if params:
            merged.update(dict(params))
        merged.update(kwargs)

        # Extraer file_name ANTES de construir TranscriptionParams: el modelo
        # tiene extra="forbid" y rechazaría un campo desconocido.
        effective_file_name: str = merged.pop("file_name", self.file_name)

        params_obj = TranscriptionParams(**merged)
        form_data = params_obj.to_form_data()

        mime_type = self._mime_type_for(effective_file_name)
        files_dict: dict[str, tuple[str, bytes, str]] = {
            "file": (effective_file_name, audio, mime_type),
        }

        return files_dict, form_data, params_obj

    @staticmethod
    def _parse_transcription_response(
        resp: httpx.Response,
        response_format: TranscriptionFormat,
    ) -> TranscriptionResponse | str:
        """
        Parse the HTTP response body from the transcription endpoint.

        The parse strategy is driven by the actual ``Content-Type`` header of
        the response rather than by the requested ``response_format``. The
        Pollinations API always returns ``application/json`` regardless
        of the requested format; this method therefore defaults to JSON parsing
        and only falls back to plain-text for responses whose Content-Type is
        not ``application/json``.

        Args:
            resp: The `httpx.Response` from the transcription endpoint.
            response_format: The output format requested (kept for forward
                compatibility; currently not used to branch the parsing logic).

        Returns:
            A `TranscriptionResponse` when the response is JSON, or a
            plain ``str`` for non-JSON content types.
        """
        content_type = resp.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            return TranscriptionResponse.model_validate(resp.json())
        # Formatos text/srt/vtt: devolver texto plano.
        return resp.text

    def transcribe_response(
        self,
        audio: bytes,
        *,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Execute a synchronous POST request to the transcription endpoint and
        return the raw HTTP response.

        Use `transcribe` to receive a parsed `TranscriptionResponse`
        or plain string directly.

        Args:
            audio: Raw audio bytes to transcribe.
            params: Optional per-call parameter overrides (may include
                ``file_name``).
            **kwargs: Additional per-call overrides.

        Returns:
            The raw `httpx.Response` from the API.

        Raises:
            PollinationsAPIError: On any non-2xx HTTP response.
        """
        files_dict, form_data, _ = self._build_multipart(audio, params, **kwargs)
        return self._http.post_multipart(
            "/v1/audio/transcriptions",
            files=files_dict,
            data=form_data,
        )

    async def atranscribe_response(
        self,
        audio: bytes,
        *,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Execute an asynchronous POST request to the transcription endpoint and
        return the raw HTTP response.

        Use `atranscribe` to receive a parsed `TranscriptionResponse`
        or plain string directly.

        Args:
            audio: Raw audio bytes to transcribe.
            params: Optional per-call parameter overrides (may include
                ``file_name``).
            **kwargs: Additional per-call overrides.

        Returns:
            The raw `httpx.Response` from the API.

        Raises:
            PollinationsAPIError: On any non-2xx HTTP response.
        """
        files_dict, form_data, _ = self._build_multipart(audio, params, **kwargs)
        return await self._http.apost_multipart(
            "/v1/audio/transcriptions",
            files=files_dict,
            data=form_data,
        )

    def transcribe(
        self,
        audio: bytes,
        *,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TranscriptionResponse | str:
        """
        Transcribe audio synchronously and return a parsed response.

        The return type depends on the effective ``response_format``:

        - ``"json"`` / ``"verbose_json"``: returns a
          `TranscriptionResponse`. For ``"verbose_json"``, additional
          fields (segments, words, language, duration, etc.) are accessible
          via ``result.model_extra``.
        - ``"text"`` / ``"srt"`` / ``"vtt"``: returns the raw transcription
          string as delivered by the API.

        Args:
            audio: Raw audio bytes to transcribe.
            params: Optional per-call parameter overrides (may include
                ``file_name``).
            **kwargs: Additional per-call overrides.

        Returns:
            A `TranscriptionResponse` for JSON-based formats, or a
            plain ``str`` for text-based formats.

        Raises:
            PollinationsAPIError: On any non-2xx HTTP response.
        """
        files_dict, form_data, params_obj = self._build_multipart(audio, params, **kwargs)
        resp = self._http.post_multipart(
            "/v1/audio/transcriptions",
            files=files_dict,
            data=form_data,
        )
        return self._parse_transcription_response(resp, params_obj.response_format)

    async def atranscribe(
        self,
        audio: bytes,
        *,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TranscriptionResponse | str:
        """
        Transcribe audio asynchronously and return a parsed response.

        The return type depends on the effective ``response_format``:

        - ``"json"`` / ``"verbose_json"``: returns a
          `TranscriptionResponse`. For ``"verbose_json"``, additional
          fields (segments, words, language, duration, etc.) are accessible
          via ``result.model_extra``.
        - ``"text"`` / ``"srt"`` / ``"vtt"``: returns the raw transcription
          string as delivered by the API.

        Args:
            audio: Raw audio bytes to transcribe.
            params: Optional per-call parameter overrides (may include
                ``file_name``).
            **kwargs: Additional per-call overrides.

        Returns:
            A `TranscriptionResponse` for JSON-based formats, or a
            plain ``str`` for text-based formats.

        Raises:
            PollinationsAPIError: On any non-2xx HTTP response.
        """
        files_dict, form_data, params_obj = self._build_multipart(audio, params, **kwargs)
        resp = await self._http.apost_multipart(
            "/v1/audio/transcriptions",
            files=files_dict,
            data=form_data,
        )
        return self._parse_transcription_response(resp, params_obj.response_format)

    def invoke(
        self, input: bytes, config: Any | None = None, **kwargs: Any
    ) -> TranscriptionResponse | str:
        """
        LangChain-compatible synchronous invocation method.

        Implements the Runnable duck-typing interface expected by LangChain
        v1.2.x+. The ``config`` parameter is accepted for interface
        compatibility but is not used internally.

        Args:
            input: Raw audio bytes to transcribe.
            config: Optional LangChain ``RunnableConfig`` (unused).
            **kwargs: Additional transcription parameters forwarded to
                `transcribe`.

        Returns:
            A `TranscriptionResponse` or plain ``str`` depending on
            ``response_format``.
        """
        _ = config
        return self.transcribe(input, **kwargs)

    async def ainvoke(
        self, input: bytes, config: Any | None = None, **kwargs: Any
    ) -> TranscriptionResponse | str:
        """
        LangChain-compatible asynchronous invocation method.

        Implements the async Runnable duck-typing interface expected by
        LangChain v1.2.x+. The ``config`` parameter is accepted for interface
        compatibility but is not used internally.

        Args:
            input: Raw audio bytes to transcribe.
            config: Optional LangChain ``RunnableConfig`` (unused).
            **kwargs: Additional transcription parameters forwarded to
                `atranscribe`.

        Returns:
            A `TranscriptionResponse` or plain ``str`` depending on
            ``response_format``.
        """
        _ = config
        return await self.atranscribe(input, **kwargs)
