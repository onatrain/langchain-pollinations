"""
This module provides the ImagePollinations class and associated utilities.
It facilitates image and video generation through the Pollinations.ai API.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from urllib.parse import quote

import httpx
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
    "imagen-4",
    "grok-video",
    "ltx-2",
]


Quality = Literal["low", "medium", "high", "hd"]


class ImagePromptParams(BaseModel):
    """
    Data structure representing the official query parameters for image generation.
    Includes configuration for model, dimensions, seed, and model-specific options.
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
        """
        Convert the parameter model into a dictionary suitable for URL query strings.

        Returns:
            A dictionary containing active parameters with appropriate aliases.
        """
        return self.model_dump(by_alias=True, exclude_none=True)


class ImagePollinations(BaseModel):
    """
    Configurable wrapper for the Pollinations image generation endpoint.
    Supports both synchronous and asynchronous operations with LangChain integration.
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
        """
        Initialize the image client with provided configuration and credentials.

        Args:
            **data: Configuration parameters including api_key, base_url, and defaults.
        """
        super().__init__(**data)
        auth = AuthConfig.from_env_or_value(self.api_key)
        self._http = PollinationsHttpClient(
            config=HttpConfig(base_url=self.base_url, timeout_s=self.timeout_s),
            api_key=auth.api_key,
        )

    def with_params(self, **overrides: Any) -> ImagePollinations:
        """
        Create a new instance with updated default parameters.

        Args:
            **overrides: Parameter values to override in the new instance.

        Returns:
            A new ImagePollinations instance with merged configuration.
        """
        merged = self.model_dump(by_alias=True, exclude_none=False)
        merged.update(overrides)
        # api_key se excluye del dump; lo reinyectamos
        merged["api_key"] = self.api_key
        return ImagePollinations(**merged)

    def _defaults_dict(self) -> dict[str, Any]:
        """
        Extract the current instance configuration into a clean dictionary.

        Returns:
            A dictionary containing only the parameters explicitly configured.
        """
        out: dict[str, Any] = {}
        for k in (
            "model",
            "width",
            "height",
            "seed",
            "enhance",
            "negative_prompt",
            "safe",
            "quality",
            "image",
            "transparent",
            "duration",
            "aspectRatio",
            "audio",
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
        """
        Build and validate the final query parameters for an API request.

        Args:
            params: Optional dictionary of parameters to include.
            **kwargs: Additional parameters passed as keyword arguments.

        Returns:
            A validated dictionary of query parameters.
        """
        merged: dict[str, Any] = {}
        merged.update(self._defaults_dict())
        if params:
            merged.update(dict(params))
        if kwargs:
            merged.update(kwargs)

        validated = ImagePromptParams(**merged)
        return validated.to_query()

    def generate_response(
        self, prompt: str, *, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> httpx.Response:
        """
        Execute a synchronous request to the image generation endpoint.

        Args:
            prompt: The text description of the image to generate.
            params: Optional dictionary of query parameters.
            **kwargs: Additional parameters for the request.

        Returns:
            The raw HTTP response from the server.
        """
        encoded = quote(prompt, safe="")
        q = self._build_query(params, **kwargs)
        return self._http.get(f"/image/{encoded}", params=q)

    async def agenerate_response(
        self, prompt: str, *, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> httpx.Response:
        """
        Execute an asynchronous request to the image generation endpoint.

        Args:
            prompt: The text description of the image to generate.
            params: Optional dictionary of query parameters.
            **kwargs: Additional parameters for the request.

        Returns:
            The raw HTTP response from the server.
        """
        encoded = quote(prompt, safe="")
        q = self._build_query(params, **kwargs)
        return await self._http.aget(f"/image/{encoded}", params=q)

    def generate(
        self, prompt: str, *, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> bytes:
        """
        Generate an image synchronously and return the raw bytes.

        Args:
            prompt: The text description of the image to generate.
            params: Optional dictionary of query parameters.
            **kwargs: Additional parameters for the request.

        Returns:
            The binary content of the generated image or video.
        """
        return self.generate_response(prompt, params=params, **kwargs).content

    async def agenerate(
        self, prompt: str, *, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> bytes:
        """
        Generate an image asynchronously and return the raw bytes.

        Args:
            prompt: The text description of the image to generate.
            params: Optional dictionary of query parameters.
            **kwargs: Additional parameters for the request.

        Returns:
            The binary content of the generated image or video.
        """
        r = await self.agenerate_response(prompt, params=params, **kwargs)
        return r.content

    # Runnable-like (para ecosistema LC)
    def invoke(self, input: str, config: Any | None = None, **kwargs: Any) -> bytes:
        """
        LangChain-compatible synchronous invocation method.

        Args:
            input: The text prompt for generation.
            config: Optional LangChain configuration (unused).
            **kwargs: Additional generation parameters.

        Returns:
            The binary content of the generated media.
        """
        _ = config
        return self.generate(input, **kwargs)

    async def ainvoke(self, input: str, config: Any | None = None, **kwargs: Any) -> bytes:
        """
        LangChain-compatible asynchronous invocation method.

        Args:
            input: The text prompt for generation.
            config: Optional LangChain configuration (unused).
            **kwargs: Additional generation parameters.

        Returns:
            The binary content of the generated media.
        """
        _ = config
        return await self.agenerate(input, **kwargs)
