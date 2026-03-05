from __future__ import annotations

import pathlib

import dotenv

from langchain_pollinations.tts import TTSPollinations

dotenv.load_dotenv()

OUTPUT_FILE = pathlib.Path("three_audios.mp3")

# Tres clips: (modelo, voz, texto)
CLIPS: list[tuple[str, str, str]] = [
    (
        "elevenlabs",
        "rachel",
        "The ancient lighthouse guided sailors through stormy nights.",
    ),
    (
        "elevenlabs",
        "bella",
        "Every single word she spoke carried unexpected meaning.",
    ),
    (
        "scribe",
        "adam",
        "Mountains remember every storm that has ever passed.",
    ),
]

# Limpiar el archivo de salida si existe de una corrida anterior.
OUTPUT_FILE.write_bytes(b"")

# Cliente base compartido; cada clip sobreescribe model y voice vía with_params().
base = TTSPollinations(response_format="mp3")

for index, (model, voice, text) in enumerate(CLIPS, start=1):
    print(f"[{index}/3] modelo={model!r}  voz={voice!r}")
    print(f"       texto: {text!r}")

    # Configurar el cliente para este clip específico.
    clip_client = base.with_params(model=model, voice=voice)

    # Rotar entre los tres métodos de generación para demostrar equivalencia.
    if index == 1:
        # Método directo: retorna bytes.
        audio: bytes = clip_client.generate(text)

    elif index == 2:
        # Vía generate_response(): acceso al httpx.Response completo antes de .content.
        response = clip_client.generate_response(text)
        print(f"       content-type: {response.headers.get('content-type', 'n/a')}")
        audio = response.content

    else:
        # Interfaz LangChain Runnable: invoke() es alias de generate().
        audio = clip_client.invoke(text)

    # Acumular en el archivo de salida (modo append binario).
    with OUTPUT_FILE.open("ab") as fh:
        fh.write(audio)

    size_kb = len(audio) / 1024
    total_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"       clip: {size_kb:.1f} KB  |  acumulado: {total_kb:.1f} KB\n")

print(f"Archivo final: {OUTPUT_FILE}  ({OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")
