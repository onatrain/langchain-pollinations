"""
Genera una conversación de dos amigos mediante TTSPollinations —alternando
voces por orador—, acumula el audio en memoria como WAV unificado y lo
transcribe con STTPollinations, mostrando el resultado formateado en el terminal.
"""

from __future__ import annotations

import io
import dotenv
import struct
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
        "preguntarte si usarías calculadora para contabilidad. ¡La "
        "respuesta es obvia!",
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


# Concatenación WAV sin dependencias externas

def _build_wav_header(
    nchannels: int,
    sampwidth: int,
    framerate: int,
    data_size: int,
) -> bytes:
    """
    Build a minimal 44-byte WAV/RIFF header for PCM audio.

    Constructs the header manually with ``struct.pack`` to avoid any
    dependency on ``wave.Wave_write``, whose close/patch path in CPython
    3.12 can raise ``struct.error`` when the WAV chunks returned by the
    TTS API carry an ``nframes`` field inconsistent with the actual PCM
    payload size.

    The resulting header is compliant with the canonical RIFF PCM spec:
    ``RIFF`` chunk → ``WAVE`` + ``fmt `` sub-chunk (16 bytes, PCM=1) +
    ``data`` sub-chunk descriptor.

    Args:
        nchannels: Number of audio channels (1 = mono, 2 = stereo).
        sampwidth: Sample width in bytes (e.g. 2 for 16-bit audio).
        framerate: Sample rate in Hz (e.g. 24000).
        data_size: Total size of the raw PCM data in bytes.

    Returns:
        A 44-byte bytes object containing the complete WAV header.
    """
    byte_rate   = nchannels * framerate * sampwidth
    block_align = nchannels * sampwidth
    bits        = sampwidth * 8

    # Formato RIFF canónico (44 bytes):
    #   "RIFF" <riff_size:L> "WAVE"
    #   "fmt " <16:L> <PCM=1:H> <nch:H> <rate:L> <brate:L> <align:H> <bits:H>
    #   "data" <data_size:L>
    return struct.pack(
        "<4sL4s4sLHHLLHH4sL",
        b"RIFF",
        36 + data_size,   # tamaño del chunk RIFF = cabecera fmt (36) + datos
        b"WAVE",
        b"fmt ",
        16,               # tamaño del sub-chunk fmt para PCM
        1,                # audio format: 1 = PCM lineal
        nchannels,
        framerate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size,
    )


def _concatenate_wav_chunks(chunks: list[bytes]) -> bytes:
    """
    Merge a list of WAV byte chunks into a single valid WAV byte stream.

    Uses ``wave.open`` **only for reading** — extracting PCM frames and audio
    parameters from each chunk. The output WAV header is built manually via
    :func:`_build_wav_header` with ``struct.pack``, bypassing
    ``wave.Wave_write`` entirely and avoiding the CPython 3.12 close/patch
    path that raises ``struct.error`` on WAV files with inconsistent
    ``nframes`` headers (common in TTS API responses).

    Args:
        chunks: Non-empty list of raw WAV bytes, one per dialogue line.
            All chunks must share the same audio parameters (channels,
            sample width, frame rate), which is guaranteed when all are
            produced by the same TTS model.

    Returns:
        A single well-formed WAV byte stream covering all dialogue lines
        in order, ready to be sent to a transcription endpoint.

    Raises:
        ValueError: If ``chunks`` is empty.
        wave.Error: If any chunk cannot be opened as a valid WAV file.
    """
    if not chunks:
        raise ValueError("chunks must contain at least one WAV byte string.")

    params = None
    raw_frames_parts: list[bytes] = []

    for chunk in chunks:
        # Solo se usa wave en modo lectura: no hay Wave_write ni close/patch.
        with wave.open(io.BytesIO(chunk), "rb") as in_wav:
            if params is None:
                params = in_wav.getparams()
            # readframes() devuelve los bytes PCM puros, sin ningún header.
            raw_frames_parts.append(in_wav.readframes(in_wav.getnframes()))

    all_frames: bytes = b"".join(raw_frames_parts)
    header = _build_wav_header(
        nchannels=params.nchannels,
        sampwidth=params.sampwidth,
        framerate=params.framerate,
        data_size=len(all_frames),
    )
    return header + all_frames


def _write_to_file(file_name: str, audio_bytes: bytes) -> None:
    with open(file_name, "wb") as f:
        f.write(audio_bytes)


def generate_dialogue_audio(
    tts: TTSPollinations,
    dialogue: list[DialogueLine],
) -> bytes:
    """
    Generate TTS audio for every dialogue line and return a single WAV stream.

    Each line is synthesised individually in WAV format using the speaker's
    designated voice. The resulting chunks are merged via
    :func:`_concatenate_wav_chunks` into one continuous WAV file so that
    Whisper receives a single unambiguous audio stream covering the full
    dialogue.

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
        chunk = tts.generate(
            line.text,
            voice=line.voice,
            response_format="wav",
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

    file_name = "dialogue.wav"

    tts = TTSPollinations(model="elevenlabs")
    stt = STTPollinations(
        model="scribe",
        response_format="json",
        language="es",
        file_name=file_name,
    )

    # --- Fase 1: generación TTS ---
    _print_section(f"Generando audio TTS  ({len(DIALOGUE)} líneas · 2 voces)")
    audio_bytes = generate_dialogue_audio(tts, DIALOGUE)
    _write_to_file(file_name, audio_bytes)

    # --- Fase 2: transcripción STT ---
    _print_section("Transcribiendo con STTPollinations…")
    result = transcribe_dialogue(stt, audio_bytes)

    # --- Fase 3: presentación ---
    if isinstance(result, str):
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
