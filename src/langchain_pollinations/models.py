"""
This module provides the ModelInformation class to interact with Pollinations model endpoints.
It facilitates listing and filtering available text and image models via synchronous and asynchronous methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_pollinations._auth import AuthConfig
from langchain_pollinations._client import HttpConfig, PollinationsHttpClient

DEFAULT_BASE_URL = "https://gen.pollinations.ai"


@dataclass(slots=True)
class ModelInformation:
    """
    Handles retrieval and parsing of available models from the Pollinations API.
    Provides methods to list compatible, text-based, and image-based models.
    """
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 120.0

    _http: PollinationsHttpClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Initialize the internal HTTP client after data class instantiation.

        This method sets up the authentication and configuration for the PollinationsHttpClient.
        """
        auth = AuthConfig.from_env_or_value(self.api_key)
        self._http = PollinationsHttpClient(
            config=HttpConfig(base_url=self.base_url, timeout_s=self.timeout_s),
            api_key=auth.api_key,
        )

    @staticmethod
    def _extract_model_ids(models_data: Any) -> list[str]:
        """
        Extract model identifiers from the API response payload.

        Args:
            models_data: The raw data returned from the models endpoint, expected to be a list of dictionaries.

        Returns:
            A list of strings representing the unique identifiers of the models.
        """
        if not isinstance(models_data, list):
            return []

        model_ids: list[str] = []

        for model in models_data:
            if not isinstance(model, dict):
                continue

            # Buscar identificador en orden de prioridad
            model_id = model.get("id") or model.get("model") or model.get("name")

            if model_id and isinstance(model_id, str):
                model_ids.append(model_id)

        return model_ids

    def get_available_models(self) -> dict[str, list[str]]:
        """
        Retrieve identifiers for all available text and image models synchronously.

        Returns:
            A dictionary with "text" and "image" keys, each containing a list of model IDs.
        """
        text_ids: list[str] = []
        image_ids: list[str] = []

        # Obtener modelos de texto
        try:
            text_models = self.list_text_models()
            text_ids = self._extract_model_ids(text_models)
        except Exception:
            # Si falla, retornar lista vacía para texto
            pass

        # Obtener modelos de imagen
        try:
            image_models = self.list_image_models()
            image_ids = self._extract_model_ids(image_models)
        except Exception:
            # Si falla, retornar lista vacía para imagen
            pass

        return {
            "text": text_ids,
            "image": image_ids,
        }

    async def aget_available_models(self) -> dict[str, list[str]]:
        """
        Retrieve identifiers for all available text and image models asynchronously.

        Returns:
            A dictionary with "text" and "image" keys, each containing a list of model IDs.
        """
        text_ids: list[str] = []
        image_ids: list[str] = []

        # Obtener modelos de texto
        try:
            text_models = await self.alist_text_models()
            text_ids = self._extract_model_ids(text_models)
        except Exception:
            # Si falla, retornar lista vacía para texto
            pass

        # Obtener modelos de imagen
        try:
            image_models = await self.alist_image_models()
            image_ids = self._extract_model_ids(image_models)
        except Exception:
            # Si falla, retornar lista vacía para imagen
            pass

        return {
            "text": text_ids,
            "image": image_ids,
        }

    def list_compatible_models(self) -> dict[str, Any]:
        """
        List all OpenAI-compatible models available via the /v1/models endpoint.

        Returns:
            A dictionary containing the API response for compatible models.
        """
        return self._http.get("/v1/models").json()

    def list_text_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        """
        Fetch the full list of available text models from the /text/models endpoint.

        Returns:
            A list or dictionary containing detailed information about text models.
        """
        return self._http.get("/text/models").json()

    def list_image_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        """
        Fetch the full list of available image models from the /image/models endpoint.

        Returns:
            A list or dictionary containing detailed information about image models.
        """
        return self._http.get("/image/models").json()

    async def alist_compatible_models(self) -> dict[str, Any]:
        """
        Asynchronously list all OpenAI-compatible models available via the /v1/models endpoint.

        Returns:
            A dictionary containing the API response for compatible models.
        """
        return (await self._http.aget("/v1/models")).json()

    async def alist_text_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        """
        Asynchronously fetch the full list of available text models from the /text/models endpoint.

        Returns:
            A list or dictionary containing detailed information about text models.
        """
        return (await self._http.aget("/text/models")).json()

    async def alist_image_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        """
        Asynchronously fetch the full list of available image models from the /image/models endpoint.

        Returns:
            A list or dictionary containing detailed information about image models.
        """
        return (await self._http.aget("/image/models")).json()
