"""
Shared audio model catalog for the Pollinations AI audio endpoints.

Provides the module-level cache, threading primitives, loader function, and
the ``AudioModelId`` type alias used by both `tts` and `stt` to
validate audio model identifiers against the live ``/audio/models`` catalog.

All exported names are designed for explicit per-name imports so that unit
tests can patch or reset individual pieces in isolation::

    from langchain_pollinations._audio_catalog import (
        _audio_model_ids_cache,
        _audio_model_ids_loaded,
        _FALLBACK_AUDIO_MODEL_IDS,
        _load_audio_model_ids,
        AudioModelId,
    )
"""

from __future__ import annotations

import threading

# Alias de tipo: documenta intención sin imponer restricción estática.
# Usar str permite tolerar modelos nuevos del API sin ValidationError.
AudioModelId = str

# Modelos conocidos al momento del release; sirven de fallback cuando el API
# no responde. Incluye tanto modelos TTS como STT porque /audio/models los
# devuelve juntos en un único endpoint.
_FALLBACK_AUDIO_MODEL_IDS: list[str] = [
    "openai-audio",
    "tts-1",
    "elevenlabs",
    "elevenmusic",
    "whisper-large-v3",
    "whisper-1",
    "scribe",
]

# Caché mutable del catálogo. Se inicializa con el fallback y se actualiza
# en la primera llamada exitosa a _load_audio_model_ids().
_audio_model_ids_cache: list[str] = list(_FALLBACK_AUDIO_MODEL_IDS)

# Primitivas de sincronización: la carga remota se ejecuta a lo sumo una vez
# por proceso, incluso en contextos multi-hilo.
_audio_model_ids_lock: threading.Lock = threading.Lock()
_audio_model_ids_loaded: bool = False


def _load_audio_model_ids(
    api_key: str | None = None,
    *,
    force: bool = False,
) -> list[str]:
    """
    Fetch the list of available audio model IDs from the Pollinations API and
    update the module-level cache.

    The remote call is made at most once per process lifetime. Subsequent calls
    return the cached list immediately unless ``force=True`` is passed. If the
    API call fails for any reason (network error, missing key, etc.), the cache
    retains its current value without raising an exception.

    This function is safe for concurrent use. It uses a double-checked locking
    pattern to avoid both race conditions and unnecessary lock contention after
    the catalog has been loaded.

    Args:
        api_key: API key forwarded to ``ModelInformation``. When ``None`` the
            value is resolved from the ``POLLINATIONS_API_KEY`` environment
            variable. If neither is available the call fails silently and the
            fallback list is kept.
        force: When ``True``, bypass the one-shot guard and re-fetch from the
            API regardless of whether a previous successful call was already
            made. Useful for explicit catalog refreshes at runtime.

    Returns:
        A copy of the current (possibly freshly updated) audio model ID list.
    """
    global _audio_model_ids_cache, _audio_model_ids_loaded

    # Lectura rápida fuera del lock: evita contención en el caso común
    # (catálogo ya cargado, force=False).
    if _audio_model_ids_loaded and not force:
        return list(_audio_model_ids_cache)

    with _audio_model_ids_lock:
        # Segunda verificación dentro del lock (double-checked locking) para
        # descartar la carrera entre hilos que pasaron el primer if.
        if _audio_model_ids_loaded and not force:
            return list(_audio_model_ids_cache)

        try:
            # Import local para evitar importaciones circulares al nivel de módulo
            # (_audio_catalog.py y models.py comparten el mismo paquete).
            from langchain_pollinations.models import ModelInformation  # noqa: PLC0415

            info = ModelInformation(api_key=api_key)
            ids: list[str] = info.get_available_models().get("audio", [])

            if ids:
                _audio_model_ids_cache = ids

        except Exception:
            # Cualquier fallo deja el caché intacto. Se marca como intentado
            # igualmente para no reintentar en cada instantiación.
            pass

        # Marcar como intentado siempre: evita martillar el API en entornos
        # sin conectividad. Usar force=True para forzar un reintento explícito.
        _audio_model_ids_loaded = True

    return list(_audio_model_ids_cache)
