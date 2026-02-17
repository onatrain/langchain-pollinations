from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_pollinations._auth import AuthConfig
from langchain_pollinations._client import HttpConfig, PollinationsHttpClient

DEFAULT_BASE_URL = "https://gen.pollinations.ai"


@dataclass(slots=True)
class ModelInformation:
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 120.0

    _http: PollinationsHttpClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        auth = AuthConfig.from_env_or_value(self.api_key)
        self._http = PollinationsHttpClient(
            config=HttpConfig(base_url=self.base_url, timeout_s=self.timeout_s),
            api_key=auth.api_key,
        )

    @staticmethod
    def _extract_model_ids(models_data: Any) -> list[str]:
        """
        Extrae los identificadores de modelos de una respuesta de API.

        Args:
            models_data: Respuesta del endpoint (esperado: lista de dicts)

        Returns:
            Lista de identificadores de modelos (strings).
            Lista vacía si la respuesta no es válida.
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
        Obtiene los identificadores de todos los modelos disponibles.

        Combina las respuestas de /text/models y /image/models en un
        diccionario con dos listas: una para modelos de texto y otra
        para modelos de imagen.

        Returns:
            Diccionario con el formato:
            {
                "text": ["openai", "claude", "gemini", ...],
                "image": ["flux", "gptimage", "veo", ...]
            }

            Si un endpoint falla, retorna lista vacía para ese tipo.

        Example:
            >>> models = ModelInformation(apikey="your-key")
            >>> available = models.get_available_models()
            >>> print(available["text"])
            ['openai', 'claude', 'gemini', 'mistral', ...]
            >>>
            >>> # Verificar si un modelo específico está disponible
            >>> if "flux" in available["image"]:
            ...     print("Flux está disponible")
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
        Versión async de get_available_models().

        Obtiene los identificadores de todos los modelos disponibles
        de forma asíncrona.

        Returns:
            Diccionario con el formato:
            {
                "text": ["openai", "claude", "gemini", ...],
                "image": ["flux", "gptimage", "veo", ...]
            }

        Example:
            >>> import asyncio
            >>>
            >>> async def main():
            ...     models = ModelInformation(apikey="your-key")
            ...     available = await models.aget_available_models()
            ...     print(f"Modelos de texto: {len(available['text'])}")
            ...     print(f"Modelos de imagen: {len(available['image'])}")
            >>>
            >>> asyncio.run(main())
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
        return self._http.get("/v1/models").json()

    def list_text_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        return self._http.get("/text/models").json()

    def list_image_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        return self._http.get("/image/models").json()

    async def alist_compatible_models(self) -> dict[str, Any]:
        return (await self._http.aget("/v1/models")).json()

    async def alist_text_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        return (await self._http.aget("/text/models")).json()

    async def alist_image_models(self) -> list[dict[str, Any]] | dict[str, Any]:
        return (await self._http.aget("/image/models")).json()
