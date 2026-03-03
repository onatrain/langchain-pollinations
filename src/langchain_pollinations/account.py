"""
This module provides classes for managing and retrieving Pollinations account information.
It includes support for profile details, balance checks, API keys, and detailed usage statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from langchain_pollinations._auth import AuthConfig
from langchain_pollinations._client import HttpConfig, PollinationsHttpClient

DEFAULT_BASE_URL = "https://gen.pollinations.ai"

AccountFormat = Literal["json", "csv"]
Limit1To50000 = Annotated[int, Field(ge=1, le=50000)]

# Niveles de suscripción que el backend puede devolver en el campo `tier`.
AccountTier = Literal["anonymous", "microbe", "spore", "seed", "flower", "nectar", "router"]

# Tipo de API key. Valores conocidos al momento del release; pueden ampliarse.
# Se usa str en los modelos para evitar ValidationError ante nuevos valores.
ApiKeyType = str  # known: "secret" | "publishable"

# Fuente de billing. Valores conocidos al momento del release; pueden ampliarse.
MeterSource = str  # known: "tier" | "pack" | "crypto"


class AccountProfile(BaseModel):
    """
    Typed representation of a Pollinations account profile.

    Mirrors the JSON body returned by the ``GET /account/profile`` endpoint.
    camelCase API field names are mapped to Python-friendly snake_case attributes
    via ``Field(alias=...)``. Unknown fields added by the API in future releases
    are preserved without raising a validation error (``extra="allow"``).

    All fields marked as nullable in the API spec are typed as ``Optional``
    with a default of ``None``, even though the response always includes those
    keys (they can carry a ``null`` value).

    Attributes:
        name: Display name of the account holder. ``None`` when not configured.
        email: Email address linked to the account. ``None`` when not provided.
        github_username: GitHub username linked to the account. ``None`` when
            no GitHub account is connected.
        image: Profile picture URL (e.g. the GitHub avatar URL). ``None`` when
            the account has no avatar configured.
        tier: Subscription tier of the account. Possible values in ascending
            order: ``"anonymous"``, ``"microbe"``, ``"spore"``, ``"seed"``,
            ``"flower"``, ``"nectar"``, ``"router"``.
        created_at: ISO 8601 UTC timestamp recording when the account was created.
        next_reset_at: ISO 8601 UTC timestamp of the next daily Pollen balance
            reset. ``None`` for accounts without a scheduled reset cycle.
    """

    # extra="allow" para tolerar nuevos campos del backend sin romper la validación.
    # populate_by_name=True permite usar tanto el alias camelCase como el nombre
    # snake_case al construir el modelo manualmente en tests.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: Optional[str] = None
    email: Optional[str] = None

    # El backend devuelve "githubUsername"; exponemos github_username en Python.
    github_username: Optional[str] = Field(default=None, alias="githubUsername")

    # Campo añadido en API v0.3.0: URL de la foto de perfil (avatar de GitHub u otro proveedor).
    image: Optional[str] = None

    # Sin default: el backend lo marca como required y nunca debería llegar como null.
    tier: AccountTier

    # Sin default: el backend lo marca como required; ISO 8601 string (no datetime) para
    # mantener consistencia con el resto del módulo y evitar fricción de zona horaria.
    created_at: str = Field(alias="createdAt")

    # Puede ser null para cuentas sin ciclo de reset programado.
    next_reset_at: Optional[str] = Field(default=None, alias="nextResetAt")


class AccountUsageRecord(BaseModel):
    """
    Typed representation of a single usage record from ``GET /account/usage``.

    Each instance corresponds to one entry in the ``usage`` array of the API
    response. All field names are already snake_case in the API, so no aliases
    are needed.

    Token counts use ``float`` to match the JSON ``number`` type exactly and
    avoid silent coercion errors. Logically they are non-negative integers, but
    the API spec declares them as generic numbers. Unknown fields added by the
    backend are preserved via ``extra="allow"``.

    Attributes:
        timestamp: Request timestamp in ``YYYY-MM-DD HH:mm:ss`` format.
        type: Request type string. Known values: ``"generate.image"``,
            ``"generate.text"``.
        model: Model ID used for the generation. ``None`` when not applicable.
        api_key: Masked API key identifier used for the request. ``None`` when
            the request was made without a key.
        api_key_type: Category of the API key. Known values: ``"secret"``,
            ``"publishable"``. Typed as ``str`` to tolerate future additions
            without raising a validation error.
        meter_source: Billing source that was charged. Known values: ``"tier"``,
            ``"pack"``, ``"crypto"``. Typed as ``str`` for the same reason.
        input_text_tokens: Number of text tokens in the prompt.
        input_cached_tokens: Number of prompt tokens served from cache.
        input_audio_tokens: Number of audio tokens in the prompt.
        input_image_tokens: Number of image tokens in the prompt.
        output_text_tokens: Number of text tokens in the completion.
        output_reasoning_tokens: Number of internal reasoning tokens produced
            by chain-of-thought models.
        output_audio_tokens: Number of audio tokens in the completion.
        output_image_tokens: Number of image tokens generated (1 per image).
        cost_usd: Total cost of the request in US dollars.
        response_time_ms: End-to-end latency in milliseconds. ``None`` when
            timing data is unavailable.
    """

    # extra="allow" para absorber campos de billing que el backend pueda agregar.
    model_config = ConfigDict(extra="allow")

    # --- Metadatos de la request ---
    # Presentes siempre; sin default.
    timestamp: str
    type: str  # valores conocidos: 'generate.image', 'generate.text'

    # Nullable según el spec; siempre presentes en el response pero pueden ser null.
    model: Optional[str] = None
    api_key: Optional[str] = None

    # Tipo de key: 'secret' o 'publishable'. Str para tolerar valores futuros.
    api_key_type: Optional[ApiKeyType] = None

    # Fuente de billing: 'tier', 'pack', 'crypto'. Str para tolerar valores futuros.
    meter_source: Optional[MeterSource] = None

    # --- Desglose de tokens de entrada ---
    # Required en el spec; default=0.0 para robustez ante omisiones inesperadas.
    input_text_tokens: float = 0.0
    input_cached_tokens: float = 0.0
    input_audio_tokens: float = 0.0
    input_image_tokens: float = 0.0

    # --- Desglose de tokens de salida ---
    output_text_tokens: float = 0.0
    output_reasoning_tokens: float = 0.0
    output_audio_tokens: float = 0.0
    # 1 token por imagen generada.
    output_image_tokens: float = 0.0

    # --- Métricas de costo y latencia ---
    cost_usd: float = 0.0
    # Nullable: puede estar ausente en registros históricos sin telemetría.
    response_time_ms: Optional[float] = None


class AccountUsageResponse(BaseModel):
    """
    Typed representation of the full JSON body returned by ``GET /account/usage``.

    Wraps the list of `AccountUsageRecord` objects along with the total
    count of records included in the response. Unknown top-level fields are
    preserved via ``extra="allow"``.

    Attributes:
        usage: List of individual usage records, one per API request logged.
        count: Number of records returned in this response. Reflects the
            effective ``limit`` applied to the query.
    """

    # extra="allow" para preservar campos de envelope que el backend pueda agregar.
    model_config = ConfigDict(extra="allow")

    usage: list[AccountUsageRecord]
    # El spec declara count como number; en la práctica es siempre un entero.
    count: int


class AccountUsageParams(BaseModel):
    """
    Configuration parameters for fetching detailed account usage logs.
    Supports filtering by date, limit, and output format (JSON or CSV).
    """

    model_config = ConfigDict(extra="forbid")
    format: AccountFormat = "json"
    limit: Limit1To50000 = 100
    before: Optional[str] = None


class AccountUsageDailyParams(BaseModel):
    """
    Configuration parameters for fetching daily aggregated account usage.
    Allows specifying the desired output format for the daily usage report.
    """

    model_config = ConfigDict(extra="forbid")
    format: AccountFormat = "json"


@dataclass(slots=True)
class AccountInformation:
    """
    Main interface for interacting with Pollinations account-related endpoints.
    Provides synchronous and asynchronous methods to access profile, balance, and usage data.
    """

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 120.0

    _http: PollinationsHttpClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Initialize the HTTP client and authentication configuration after dataclass initialization.
        """
        auth = AuthConfig.from_env_or_value(self.api_key)
        self._http = PollinationsHttpClient(
            config=HttpConfig(base_url=self.base_url, timeout_s=self.timeout_s),
            api_key=auth.api_key,
        )

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict[str, Any] | str:
        """
        Parse the HTTP response based on its Content-Type header.

        Args:
            response: The raw HTTP response object from the client.

        Returns:
            A dictionary for JSON responses or a string for CSV/text responses.
        """
        content_type = response.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            return response.json()
        elif "text/csv" in content_type or "text/plain" in content_type:
            return response.text
        else:
            try:
                return response.json()
            except Exception:
                return response.text

    def get_profile(self) -> AccountProfile:
        """
        Retrieve the account profile information synchronously.

        Returns:
            An `AccountProfile` instance populated with the data returned
            by the ``GET /account/profile`` endpoint, including ``name``,
            ``email``, ``github_username``, ``image``, ``tier``, ``created_at``,
            and ``next_reset_at``.
        """
        # El endpoint responde con camelCase; AccountProfile resuelve las aliases.
        data = self._http.get("/account/profile").json()
        return AccountProfile.model_validate(data)

    def get_balance(self) -> dict[str, Any]:
        """
        Check the current account balance synchronously.

        Returns:
            A dictionary containing balance information and currency details.
        """
        return self._http.get("/account/balance").json()

    def get_key(self) -> dict[str, Any]:
        """
        Retrieve the current API key information for the account.

        Returns:
            A dictionary containing the API key and its metadata.
        """
        return self._http.get("/account/key").json()

    def get_usage(self, params: AccountUsageParams | None = None) -> AccountUsageResponse | str:
        """
        Fetch detailed account usage records synchronously.

        When ``params.format`` is ``"json"`` (the default), the response is
        validated and returned as an `AccountUsageResponse` instance,
        providing typed access to each `AccountUsageRecord` in the
        ``usage`` list and the total ``count``.

        When ``params.format`` is ``"csv"``, the raw CSV text is returned as a
        plain string, exactly as the API delivers it.

        Args:
            params: Optional parameters controlling format (``"json"`` or
                ``"csv"``), record limit (1–50 000), and a ``before`` cursor
                for pagination.

        Returns:
            An `AccountUsageResponse` for JSON responses, or a ``str``
            for CSV responses.
        """
        q = (params or AccountUsageParams()).model_dump(exclude_none=True)
        response = self._http.get("/account/usage", params=q)
        parsed = self._parse_response(response)
        # Si _parse_response devolvió un dict (formato JSON), construimos el modelo tipado.
        if isinstance(parsed, dict):
            return AccountUsageResponse.model_validate(parsed)
        # Formato CSV: devolver el string sin modificar.
        return parsed

    def get_usage_daily(
        self, params: AccountUsageDailyParams | None = None
    ) -> dict[str, Any] | str:
        """
        Fetch daily aggregated account usage statistics synchronously.

        Args:
            params: Optional parameters for formatting the daily usage data.

        Returns:
            The daily usage statistics as a JSON dictionary or CSV string.
        """
        q = (params or AccountUsageDailyParams()).model_dump(exclude_none=True)
        response = self._http.get("/account/usage/daily", params=q)
        return self._parse_response(response)

    async def aget_profile(self) -> AccountProfile:
        """
        Retrieve the account profile information asynchronously.

        Returns:
            An `AccountProfile` instance populated with the data returned
            by the ``GET /account/profile`` endpoint, including ``name``,
            ``email``, ``github_username``, ``image``, ``tier``, ``created_at``,
            and ``next_reset_at``.
        """
        # El endpoint responde con camelCase; AccountProfile resuelve las aliases.
        response = await self._http.aget("/account/profile")
        return AccountProfile.model_validate(response.json())

    async def aget_balance(self) -> dict[str, Any]:
        """
        Check the current account balance asynchronously.

        Returns:
            A dictionary containing balance information.
        """
        response = await self._http.aget("/account/balance")
        return response.json()

    async def aget_key(self) -> dict[str, Any]:
        """
        Retrieve the current API key information asynchronously.

        Returns:
            A dictionary containing API key details.
        """
        response = await self._http.aget("/account/key")
        return response.json()

    async def aget_usage(
        self, params: AccountUsageParams | None = None
    ) -> AccountUsageResponse | str:
        """
        Fetch detailed account usage records asynchronously.

        When ``params.format`` is ``"json"`` (the default), the response is
        validated and returned as an `AccountUsageResponse` instance,
        providing typed access to each `AccountUsageRecord` in the
        ``usage`` list and the total ``count``.

        When ``params.format`` is ``"csv"``, the raw CSV text is returned as a
        plain string, exactly as the API delivers it.

        Args:
            params: Optional parameters controlling format (``"json"`` or
                ``"csv"``), record limit (1–50 000), and a ``before`` cursor
                for pagination.

        Returns:
            An `AccountUsageResponse` for JSON responses, or a ``str``
            for CSV responses.
        """
        q = (params or AccountUsageParams()).model_dump(exclude_none=True)
        response = await self._http.aget("/account/usage", params=q)
        parsed = self._parse_response(response)
        # Si _parse_response devolvió un dict (formato JSON), construimos el modelo tipado.
        if isinstance(parsed, dict):
            return AccountUsageResponse.model_validate(parsed)
        # Formato CSV: devolver el string sin modificar.
        return parsed

    async def aget_usage_daily(
        self, params: AccountUsageDailyParams | None = None
    ) -> dict[str, Any] | str:
        """
        Fetch daily aggregated account usage statistics asynchronously.

        Args:
            params: Optional parameters for formatting.

        Returns:
            The daily usage statistics as a JSON dictionary or CSV string.
        """
        q = (params or AccountUsageDailyParams()).model_dump(exclude_none=True)
        response = await self._http.aget("/account/usage/daily", params=q)
        return self._parse_response(response)
