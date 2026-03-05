from __future__ import annotations

import asyncio
import pathlib
import time

import dotenv

from langchain_pollinations.tts import TTSPollinations

dotenv.load_dotenv()

OUTPUT_FILE = pathlib.Path("three_songs.mp3")

DURATION_S = 60.0  # un minuto por canción

# Tres canciones: (modelo, descripción)
SONGS: list[tuple[str, str]] = [
    (
        "elevenmusic",
        "Melancholic acoustic guitar ballad with soft piano in a minor key.",
    ),
    (
        "elevenmusic",
        "Upbeat retro synthwave track with driving bass and arpeggiated chords.",
    ),
    (
        "elevenmusic",
        "Cinematic orchestral piece building from strings to full ensemble.",
    ),
]


async def generate_song(
    index: int,
    model: str,
    description: str,
    api_key_env: str,
) -> tuple[int, bytes]:
    """
    Generate a single music track asynchronously.

    Args:
        index: 1-based position used only for logging.
        model: Audio model identifier (e.g. ``"elevenmusic"``).
        description: Natural-language description of the desired music.
        api_key_env: API key resolved from the environment at call time.

    Returns:
        A tuple of (index, audio_bytes) so results can be sorted after gather.
    """
    client = TTSPollinations(
        api_key=api_key_env,
        model=model,
        response_format="mp3",
        duration=DURATION_S,
        instrumental=True,
    )

    print(f"  [{index}/3] iniciando  modelo={model!r}")
    t0 = time.perf_counter()

    # Alternar entre agenerate() y ainvoke() para demostrar equivalencia.
    if index % 2 == 0:
        # ainvoke() es la interfaz LangChain Runnable asíncrona.
        audio = await client.ainvoke(description)
    else:
        # agenerate() es la interfaz nativa del módulo.
        audio = await client.agenerate(description)

    elapsed = time.perf_counter() - t0
    print(f"  [{index}/3] completado  {elapsed:.1f}s  {len(audio) / 1024:.0f} KB")

    return index, audio


async def main() -> None:
    """
    Orchestrate concurrent song generation and write the concatenated output.
    """
    import os

    api_key = os.getenv("POLLINATIONS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "POLLINATIONS_API_KEY no encontrada. "
            "Agrega la variable al archivo .env o al entorno."
        )

    print(f"Generando {len(SONGS)} canciones en paralelo ({DURATION_S:.0f}s c/u)...\n")
    wall_start = time.perf_counter()

    # Lanzar las tres generaciones de forma concurrente.
    tasks = [
        generate_song(i, model, desc, api_key)
        for i, (model, desc) in enumerate(SONGS, start=1)
    ]
    results: list[tuple[int, bytes]] = await asyncio.gather(*tasks)

    wall_elapsed = time.perf_counter() - wall_start
    print(f"\nTiempo total (paralelo): {wall_elapsed:.1f}s\n")

    # Ordenar por índice para mantener el orden original de las canciones.
    results.sort(key=lambda t: t[0])

    # Limpiar el archivo de salida y acumular en orden.
    OUTPUT_FILE.write_bytes(b"")

    total_bytes = 0
    for index, audio in results:
        model, description = SONGS[index - 1]
        with OUTPUT_FILE.open("ab") as fh:
            fh.write(audio)
        total_bytes += len(audio)
        print(
            f"  [{index}/3] {model!r:14s}  {len(audio) / 1024:.0f} KB  "
            f"— {description[:55]}..."
        )

    print(
        f"\nArchivo final: {OUTPUT_FILE}"
        f"  ({total_bytes / 1024:.0f} KB total  |  {len(SONGS)} canciones)"
    )


if __name__ == "__main__":
    asyncio.run(main())
