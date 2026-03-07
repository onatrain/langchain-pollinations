"""
Genera una conversación de dos amigos mediante TTSPollinations —alternando
voces por orador—, acumula el audio en memoria como WAV concatenado y lo
transcribe con STTPollinations, mostrando el resultado formateado en el terminal.
"""

from __future__ import annotations

import dotenv
import io
import textwrap
import wave
from dataclasses import dataclass

from langchain_pollinations.stt import STTPollinations, TranscriptionResponse
from langchain_pollinations.tts import TTSPollinations


@dataclass(frozen=True)
class DialogueLine:
    """A single spoken line in the dialogue."""
    speaker: str
    voice: str
    text: str


# Dos amigos debaten el impacto de la IA en el trabajo creativo.
DIALOGUE: list[DialogueLine] = [
    DialogueLine(
        "Alex", "alloy",
        "¿Viste que ahora todo el mundo usa IA para generar música? "
        "¡Literalmente cualquiera puede sacar un disco en una tarde!",
    ),
    DialogueLine(
        "Sam", "echo",
        "¡Sí, y lo loco es que suena bien! Yo probé uno de esos modelos "
        "la semana pasada y me quedé sin palabras. Le pedí rock de los "
        "sesenta y me gustó mucho lo que produjo.",
    ),
    DialogueLine(
        "Alex", "alloy",
        "¡Exacto!. Pero ahí está el debate: ¿tiene alma ese tipo de arte? "
        "No sé, me genera conflicto.",
    ),
    DialogueLine(
        "Sam", "echo",
        "Entiendo el conflicto, pero yo creo que el alma la pone quien "
        "hace la pregunta, no quien ejecuta. El prompt es la intención "
        "creativa.",
    ),
    DialogueLine(
        "Alex", "alloy",
        "Eso está muy bien dicho. Igual que un director de cine no opera "
        "la cámara, pero es su visión la que aparece en pantalla.",
    ),
    DialogueLine(
        "Sam", "echo",
        "Claro. Y además libera tiempo para enfocarte en lo que importa: "
        "la idea. Lo técnico ya no es la barrera.",
    ),
    DialogueLine(
        "Alex", "alloy",
        "Me convenciste. ¿Y para código también lo usas? Yo empecé hace "
        "poco y siento que multiplicó mi productividad por tres.",
    ),
    DialogueLine(
        "Sam", "echo",
        "Todo el tiempo. Para mí ya no tiene sentido no usarlo. Es como "
        "preguntarte si usarías calculadora para contabilidad. La "
        "respuesta es obvia.",
    ),
]

_W = 74

_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_CYAN    = "\033[36m"
_GREEN   = "\033[32m"
_YELLOW  = "\033[33m"
_BLUE    = "\033[34m"

_SPEAKER_COLOR: dict[str, str] = {
    "Alex": _CYAN,
    "Sam":  _GREEN,
}


def _hr(char: str = "─") -> str:
    """Return a horizontal rule of fixed width."""
    return char * _W


def _print_header() -> None:
    """Print the top-level program banner."""
    print()
    print(_BOLD + _hr("═") + _RESET)
    title = "  🎙  TTSPollinations → STTPollinations  ·  Conversación con IA"
    print(_BOLD + title + _RESET)
    print(_BOLD + _hr("═") + _RESET)
    print()


def _print_section(title: str) -> None:
    """Print a section heading with a thin rule below."""
    print()
    print(_BOLD + _YELLOW + f"  ▸ {title}" + _RESET)
    print(_DIM + "  " + _hr() + _RESET)


def _print_tts_progress(line: DialogueLine, index: int, total: int) -> None:
    """Print a one-line progress entry for a TTS generation step."""
    color = _SPEAKER_COLOR.get(line.speaker, _RESET)
    snippet = line.text[:52] + ("…" if len(line.text) > 52 else "")
    print(
        f"  [{index:>2}/{total}]  "
        f"{color}{_BOLD}{line.speaker:<5}{_RESET}  "
        f"{_DIM}voz: {line.voice:<8}{_RESET}  "
        f"{snippet}",
        flush=True,
    )


def _print_transcription(result: TranscriptionResponse) -> None:
    """Render a TranscriptionResponse to the terminal with word-wrapping."""
    _print_section("Transcripción recibida")
    wrapped = textwrap.fill(
        result.text.strip(),
        width=_W - 4,
        initial_indent="    ",
        subsequent_indent="    ",
    )
    print(_BLUE + wrapped + _RESET)

    # Mostrar campos adicionales si los devuelve verbose_json en el futuro.
    extras = result.model_extra or {}
    if extras:
        print()
        print(_DIM + "  Campos adicionales:" + _RESET)
        for key, val in extras.items():
            print(_DIM + f"    {key}: {val}" + _RESET)


def _print_summary(audio_bytes: int, line_count: int) -> None:
    """Print a closing summary block."""
    _print_section("Resumen")
    print(f"    Líneas de diálogo generadas  : {_BOLD}{line_count}{_RESET}")
    print(f"    Audio acumulado en memoria   : {_BOLD}{audio_bytes / 1024:.1f} KB{_RESET}")
    print()
    print(_BOLD + _hr("═") + _RESET)
    print()


# Procesamiento WAV

def _concatenate_wav_chunks(chunks: list[bytes]) -> bytes:
    """
    Merge a list of WAV byte chunks into a single valid WAV byte stream.

    Each chunk is a self-contained WAV file (header + PCM frames). This
    function reads the audio parameters from the first chunk, then extracts
    the raw PCM frames from every chunk and writes them under a single shared
    WAV header, producing a file that any decoder —including Whisper— handles
    as a single continuous recording.

    Using WAV (uncompressed PCM) instead of MP3 is intentional: concatenating
    raw MP3 bytes produces multiple independent ID3/MPEG headers in sequence,
    which causes Whisper to process only the first logical file and discard
    the rest.

    Args:
        chunks: Non-empty list of raw WAV bytes, one per dialogue line.

    Returns:
        A single WAV byte stream covering all dialogue lines in order.

    Raises:
        wave.Error: If any chunk cannot be parsed as a valid WAV file.
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("chunks must contain at least one WAV byte string.")

    output = io.BytesIO()

    with wave.open(output, "wb") as out_wav:
        params_initialised = False

        for chunk in chunks:
            with wave.open(io.BytesIO(chunk), "rb") as in_wav:
                # Tomar los parámetros (canales, sample rate, etc.) del primer chunk;
                # todos los chunks provienen del mismo modelo TTS, así que son idénticos.
                if not params_initialised:
                    out_wav.setparams(in_wav.getparams())
                    params_initialised = True

                # Escribir únicamente los frames PCM, sin header adicional.
                out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))

    return output.getvalue()


def generate_dialogue_audio(
    tts: TTSPollinations,
    dialogue: list[DialogueLine],
) -> bytes:
    """
    Generate TTS audio for every dialogue line and return a single WAV stream.

    Each line is synthesised individually in WAV format with the speaker's
    designated voice. The resulting chunks are merged via
    :func:`_concatenate_wav_chunks` into one continuous WAV file so that
    Whisper receives a single unambiguous audio stream.

    Args:
        tts: A configured :class:`TTSPollinations` instance.
        dialogue: Ordered list of :class:`DialogueLine` objects to synthesise.

    Returns:
        Raw WAV bytes covering the full dialogue as a single audio stream.
    """
    chunks: list[bytes] = []
    total = len(dialogue)

    for i, line in enumerate(dialogue, start=1):
        _print_tts_progress(line, i, total)
        # Voz y formato se pasan por llamada para no fijar defaults en la instancia.
        chunk = tts.generate(
            line.text,
            voice=line.voice,
            response_format="wav",   # WAV permite concatenación limpia vía stdlib wave
        )
        chunks.append(chunk)

    print(f"\n    {_DIM}Unificando {len(chunks)} fragmentos WAV…{_RESET}", flush=True)
    return _concatenate_wav_chunks(chunks)


def transcribe_dialogue(
    stt: STTPollinations,
    audio: bytes,
) -> TranscriptionResponse | str:
    """
    Send the accumulated audio bytes to the transcription endpoint.

    Args:
        stt: A configured :class:`STTPollinations` instance.
        audio: Raw audio bytes to transcribe.

    Returns:
        A :class:`TranscriptionResponse` (for ``"json"`` format) or a plain
        ``str`` (fallback for text-based formats).
    """
    kb = len(audio) / 1024
    print(f"    Tamaño del audio enviado  : {_BOLD}{kb:.1f} KB{_RESET}", flush=True)
    return stt.transcribe(audio)


def main() -> None:
    """Orchestrate TTS generation, STT transcription, and terminal display."""
    dotenv.load_dotenv()

    _print_header()

    tts = TTSPollinations(model="elevenlabs")
    stt = STTPollinations(
        response_format="json",     # único formato activo en el backend
        language="es",              # mejora la precisión para audio en español
        file_name="bad_dialogue.wav",   # la extensión determina el MIME type del multipart
    )

    # --- Fase 1: generación TTS ---
    _print_section(f"Generando audio TTS  ({len(DIALOGUE)} líneas · 2 voces)")
    audio_bytes = generate_dialogue_audio(tts, DIALOGUE)

    # --- Fase 2: transcripción STT ---
    _print_section("Transcribiendo con STTPollinations…")
    result = transcribe_dialogue(stt, audio_bytes)

    # --- Fase 3: presentación ---
    if isinstance(result, str):
        # Fallback defensivo: el backend devolvió texto plano en lugar de JSON.
        _print_section("Transcripción (texto plano — fallback)")
        wrapped = textwrap.fill(
            result.strip(),
            width=_W - 4,
            initial_indent="    ",
            subsequent_indent="    ",
        )
        print(_BLUE + wrapped + _RESET)
        print()
    else:
        _print_transcription(result)

    _print_summary(len(audio_bytes), len(DIALOGUE))


if __name__ == "__main__":
    main()


"""
This program generated this error when tried to concatenate wav outputs generated by TTS Pollinations endpoint:

Traceback (most recent call last):
  File "/code/langchain/langchain-pollinations/examples/audio/dialogue.py", line 219, in _concatenate_wav_chunks
    out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
  File "/usr/lib/python3.12/wave.py", line 576, in writeframes
    self.writeframesraw(data)
  File "/usr/lib/python3.12/wave.py", line 565, in writeframesraw
    self._ensure_header_written(len(data))
  File "/usr/lib/python3.12/wave.py", line 606, in _ensure_header_written
    self._write_header(datasize)
  File "/usr/lib/python3.12/wave.py", line 618, in _write_header
    self._file.write(struct.pack('<L4s4sLHHLLHH4s',
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
struct.error: 'L' format requires 0 <= number <= 4294967295

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/code/langchain/langchain-pollinations/examples/audio/dialogue.py", line 330, in <module>
    main()
  File "/code/langchain/langchain-pollinations/examples/audio/dialogue.py", line 305, in main
    audio_bytes = generate_dialogue_audio(tts, DIALOGUE)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/code/langchain/langchain-pollinations/examples/audio/dialogue.py", line 262, in generate_dialogue_audio
    return _concatenate_wav_chunks(chunks)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/code/langchain/langchain-pollinations/examples/audio/dialogue.py", line 207, in _concatenate_wav_chunks
    with wave.open(output, "wb") as out_wav:
  File "/usr/lib/python3.12/wave.py", line 471, in __exit__
    self.close()
  File "/usr/lib/python3.12/wave.py", line 583, in close
    self._ensure_header_written(0)
  File "/usr/lib/python3.12/wave.py", line 606, in _ensure_header_written
    self._write_header(datasize)
  File "/usr/lib/python3.12/wave.py", line 618, in _write_header
    self._file.write(struct.pack('<L4s4sLHHLLHH4s',
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
struct.error: 'L' format requires 0 <= number <= 4294967295
"""

