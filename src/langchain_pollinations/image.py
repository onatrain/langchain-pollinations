from __future__ import annotations

import httpx

from typing import Any, Literal, Optional

from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from langchain_pollinations._auth import AuthConfig
from langchain_pollinations._client import HttpConfig, PollinationsHttpClient

DEFAULT_BASE_URL = "https://gen.pollinations.ai"


# Enum de modelos según /image/{prompt} -> query param "model" en api.json.
ImageModelId = Literal[
    "kontext",
    "nanobanana",
    "nanobanana-pro",
    "seedream",
    "seedream-pro",
    "gptimage",
    "gptimage-large",
    "flux",
    "zimage",
    "veo",
    "seedance",
    "seedance-pro",
    "wan",
    "klein",
    "klein-large",
]


Quality = Literal["low", "medium", "high", "hd"]


class ImagePromptParams(BaseModel):
    """
    Query params oficiales de GET /image/{prompt} según api.json.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model: ImageModelId = "zimage"
    width: int = Field(default=1024, ge=0)
    height: int = Field(default=1024, ge=0)

    # api.json: seed default 0, min -1, max 2147483647.
    seed: int = Field(default=0, ge=-1, le=2_147_483_647)

    enhance: bool = False
    negative_prompt: str = "worst quality, blurry"
    safe: bool = False

    # api.json: quality default "medium", enum low/medium/high/hd (gptimage only).
    quality: Quality = "medium"

    # api.json: image es string (URLs ref; múltiples separadas por coma o pipe |).
    image: Optional[str] = None

    # api.json: transparent boolean (gptimage only).
    transparent: bool = False

    # api.json: duration integer 1..10 (video models).
    duration: Optional[int] = Field(default=None, ge=1, le=10)

    # api.json: aspectRatio (camelCase) string.
    aspect_ratio: Optional[str] = Field(default=None, alias="aspectRatio")

    # api.json: audio boolean (veo only). [
    audio: bool = False

    def to_query(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


class ImagePollinations(BaseModel):
    """
    Wrapper configurable para GET /image/{prompt}.

    - Estilo "ChatOpenAI": acepta parámetros de request en el constructor.
    - Integra bien en LangChain: generate/agenerate + invoke/ainvoke + with_params().
    - Retorna bytes (imagen o video).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", populate_by_name=True)

    # Auth / transporte
    api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 120.0

    # Defaults del request
    model: Optional[ImageModelId] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    enhance: Optional[bool] = None
    negative_prompt: Optional[str] = None
    safe: Optional[bool] = None
    quality: Optional[Quality] = None
    image: Optional[str] = None
    transparent: Optional[bool] = None
    duration: Optional[int] = None
    aspect_ratio: Optional[str] = Field(default=None, alias="aspectRatio")
    audio: Optional[bool] = None

    _http: PollinationsHttpClient = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        auth = AuthConfig.from_env_or_value(self.api_key)
        self._http = PollinationsHttpClient(
            config=HttpConfig(base_url=self.base_url, timeout_s=self.timeout_s),
            api_key=auth.api_key,
        )

    # --------- API "bind/clone" ---------

    def with_params(self, **overrides: Any) -> "ImagePollinations":
        """
        Clona el cliente con nuevos defaults (sin mutar el original).
        """
        merged = self.model_dump(by_alias=True, exclude_none=False)
        merged.update(overrides)
        # api_key se excluye del dump; lo reinyectamos
        merged["api_key"] = self.api_key
        return ImagePollinations(**merged)

    # --------- Construcción/validación de query ---------

    def _defaults_dict(self) -> dict[str, Any]:
        """
        Devuelve solo los campos de params que el usuario configuró en la instancia.
        Lo no configurado queda fuera para que aplique el default del schema. [file:105]
        """
        out: dict[str, Any] = {}
        for k in (
            "model", "width", "height", "seed", "enhance", "negative_prompt", "safe",
            "quality", "image", "transparent", "duration", "aspectRatio", "audio",
        ):
            # Manejo de alias aspectRatio
            if k == "aspectRatio":
                v = self.aspect_ratio
                if v is not None:
                    out["aspectRatio"] = v
                continue

            v = getattr(self, k, None)
            if v is not None:
                out[k] = v
        return out

    def _build_query(self, params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        merged.update(self._defaults_dict())
        if params:
            merged.update(dict(params))
        if kwargs:
            merged.update(kwargs)

        validated = ImagePromptParams(**merged)
        return validated.to_query()

    # --------- Métodos principales ---------

    def generate_response(self, prompt: str, *, params: dict[str, Any] | None = None, **kwargs: Any) -> "httpx.Response":
        """
        Retorna httpx.Response para poder leer:
        - response.headers["content-type"]
        - response.status_code
        - response.url
        - etc.

        El body (imagen/video) está en response.content (bytes).
        """
        encoded = quote(prompt, safe="")
        q = self._build_query(params, **kwargs)
        return self._http.get(f"/image/{encoded}", params=q)

    async def agenerate_response(
        self, prompt: str, *, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> "httpx.Response":
        encoded = quote(prompt, safe="")
        q = self._build_query(params, **kwargs)
        return await self._http.aget(f"/image/{encoded}", params=q)

    def generate(self, prompt: str, *, params: dict[str, Any] | None = None, **kwargs: Any) -> bytes:
        return self.generate_response(prompt, params=params, **kwargs).content

    async def agenerate(self, prompt: str, *, params: dict[str, Any] | None = None, **kwargs: Any) -> bytes:
        r = await self.agenerate_response(prompt, params=params, **kwargs)
        return r.content

    # Runnable-like (para ecosistema LC)
    def invoke(self, input: str, config: Any | None = None, **kwargs: Any) -> bytes:
        _ = config
        return self.generate(input, **kwargs)

    async def ainvoke(self, input: str, config: Any | None = None, **kwargs: Any) -> bytes:
        _ = config
        return await self.agenerate(input, **kwargs)
