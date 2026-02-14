from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, cast

import pytest
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from langchain_pollinations import ChatPollinations
from langchain_pollinations._openai_compat import (
    AudioTranscript,
    ContentBlock,
    ContentBlockText,
)
from langchain_pollinations.chat import AudioConfig


# ************************************************************************
# Fixtures y Helpers
# ************************************************************************

@pytest.fixture(scope="module")
def api_key() -> str:
    """Carga la API key desde el archivo .env"""
    load_dotenv(find_dotenv())
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key:
        pytest.skip("POLLINATIONS_API_KEY no encontrada en .env")
    return key


def encode_image_to_base64(image_path: str | Path) -> str:
    """Lee una imagen local y la codifica en base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_text_from_content(content: Any) -> str:
    """
    Extrae texto del content, manejando tanto strings como listas de content_blocks.
    Usa los TypedDicts para type safety.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                # Usar TypedDict para verificar estructura
                block_typed = cast(ContentBlock, block)
                if block_typed.get("type") == "text":
                    text_block = cast(ContentBlockText, block_typed)
                    text_parts.append(text_block.get("text", ""))
        return "".join(text_parts)

    return str(content)


def extract_audio_from_message(message: AIMessage) -> AudioTranscript | None:
    """
    Extrae información de audio del mensaje usando el TypedDict AudioTranscript.
    """
    audio_data = message.additional_kwargs.get("audio")
    if isinstance(audio_data, dict):
        return cast(AudioTranscript, audio_data)
    return None


# ************************************************************************
# Test 1: Text -> Text (Generación de texto estándar)
# ************************************************************************

def test_text_to_text_generation(api_key: str):
    """
    Test 1: Generación de texto con openai-fast (2-3 párrafos).
    Verifica que el modelo retorna texto coherente y maneja content_blocks si existen.
    """
    chat = ChatPollinations(
        api_key=api_key,
        request_defaults={"model": "openai-fast", "max_tokens": 1500}
    )

    messages = [
        HumanMessage(content="Escribe 2 o 3 párrafos breves sobre la inteligencia artificial.")
    ]

    response = chat.invoke(messages)

    # Verificar tipo de respuesta
    assert isinstance(response, AIMessage)

    # Extraer texto (puede venir como string o content_blocks)
    text = extract_text_from_content(response.content)

    # Validaciones básicas
    assert len(text) > 100, "El texto generado es muy corto"
    assert "inteligencia artificial" in text.lower() or "ia" in text.lower()

    # Verificar metadatos de uso
    assert response.usage_metadata is not None
    assert response.usage_metadata["total_tokens"] > 0

    print(f"\n✓ Test 1 pasado. Texto generado ({len(text)} chars):")
    print(text[:200] + "...")


# ************************************************************************
# Test 2: Text -> Audio (Generación de audio/TTS)
# ************************************************************************

def test_text_to_audio_generation(api_key: str):
    """
    Test 2: Generación de audio con openai-audio (5-6 palabras).
    Verifica que el modelo retorna audio con transcript y data en base64.
    """
    chat = ChatPollinations(
        api_key=api_key,
        request_defaults={
            "model": "openai-audio",
            "modalities": ["text", "audio"],
            "audio": AudioConfig(voice="alloy", format="mp3")
        }
    )

    messages = [
        HumanMessage(content="Hola, buenos días amigos.")
    ]

    response = chat.invoke(messages)

    # Verificar respuesta
    assert isinstance(response, AIMessage)

    # Extraer audio usando TypedDict
    audio = extract_audio_from_message(response)
    assert audio is not None, "No se recibió audio en la respuesta"

    # Validar estructura de audio
    assert "transcript" in audio or "data" in audio, "Audio no tiene transcript ni data"

    if "transcript" in audio:
        transcript = audio["transcript"]
        assert isinstance(transcript, str)
        assert len(transcript) > 0
        print(f"\n✓ Test 2 pasado. Transcript: '{transcript}'")

    if "data" in audio:
        audio_data = audio["data"]
        assert isinstance(audio_data, str)
        assert len(audio_data) > 100, "Audio data parece muy corto"
        print(f"  Audio data (base64): {len(audio_data)} chars")

    # Guardar audio para usar en tests siguientes
    pytest.test_audio_data = audio.get("data", "")
    pytest.test_audio_transcript = audio.get("transcript", "")


# ************************************************************************
# Test 3: Text + Audio -> Text (Transcripción de audio)
# ************************************************************************

def test_audio_transcription(api_key: str):
    """
    Test 3: Transcripción de audio usando openai-audio.
    Envía el audio generado en test 2 y solicita transcripción.
    """
    # Obtener audio del test anterior
    if not hasattr(pytest, "test_audio_data") or not pytest.test_audio_data:
        pytest.skip("Test 2 no generó audio, saltando test 3")

    chat = ChatPollinations(
        api_key=api_key,
        request_defaults={"model": "openai-audio"}
    )

    # Enviar audio para transcripción
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "Transcribe este audio:"},
            {"type": "input_audio", "input_audio": {"data": pytest.test_audio_data, "format": "mp3"}}
        ])
    ]

    response = chat.invoke(messages)

    # Extraer texto
    text = extract_text_from_content(response.content)

    # Validaciones
    assert len(text) > 0, "No se recibió transcripción"
    print(f"\n✓ Test 3 pasado. Transcripción: '{text}'")

    # Verificar similitud con transcript original (si existe)
    if hasattr(pytest, "test_audio_transcript") and pytest.test_audio_transcript:
        original = pytest.test_audio_transcript.lower()
        transcribed = text.lower()
        # Al menos algunas palabras deben coincidir
        original_words = set(original.split())
        transcribed_words = set(transcribed.split())
        common = original_words & transcribed_words
        assert len(common) > 0, f"Transcripción muy diferente del original. Original: {original}, Transcrito: {transcribed}"


# ************************************************************************
# Test 4: Text + Audio -> Text (Pregunta sobre audio)
# ************************************************************************

def test_audio_question_answering(api_key: str):
    """
    Test 4: Hacer una pregunta sobre el contenido de un audio.
    """
    if not hasattr(pytest, "test_audio_data") or not pytest.test_audio_data:
        pytest.skip("Test 2 no generó audio, saltando test 4")

    chat = ChatPollinations(
        api_key=api_key,
        request_defaults={"model": "openai-audio"}
    )

    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "¿De qué idioma es este audio y qué dice aproximadamente?"},
            {"type": "input_audio", "input_audio": {"data": pytest.test_audio_data, "format": "mp3"}}
        ])
    ]

    response = chat.invoke(messages)
    text = extract_text_from_content(response.content)

    assert len(text) > 10, "Respuesta muy corta"
    assert "español" in text.lower() or "spanish" in text.lower(), "No identificó el idioma correctamente"

    print(f"\n✓ Test 4 pasado. Respuesta: '{text}'")


# ************************************************************************
# Test 5: Text + Image (local) -> Text (Visión)
# ************************************************************************

def test_vision_local_image(api_key: str):
    """
    Test 5: Descripción de imagen local usando modelo openai.
    """
    # Buscar imagen
    image_path = Path(__file__).parent.parent.parent / "assets/doki.png"
    if not image_path.exists():
        pytest.skip(f"Imagen local no encontrada en {image_path}")

    # Codificar imagen
    image_base64 = encode_image_to_base64(image_path)

    chat = ChatPollinations(
        api_key=api_key,
        request_defaults={"model": "openai", "max_tokens": 1300}
    )

    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "Describe esta imagen en detalle."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ])
    ]

    response = chat.invoke(messages)
    text = extract_text_from_content(response.content)

    assert len(text) > 20, "Descripción muy corta"
    print(f"\n✓ Test 5 pasado. Descripción de imagen local:")
    print(text[:300])


# ************************************************************************
# Test 6: Text + Image (remote) -> Text (Visión)
# ************************************************************************

def test_vision_remote_image_single(api_key: str):
    """
    Test 6: Descripción de imagen remota usando modelo openai.
    Usa IMAGELOCATION_1.
    """
    IMAGE_URL = "https://i.ibb.co/CpZvrgGw/doki1.jpg"

    chat = ChatPollinations(
        api_key=api_key,
        request_defaults={"model": "openai", "max_tokens": 1300}
    )

    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "¿Qué ves en esta imagen? Descríbela."},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}}
        ])
    ]

    response = chat.invoke(messages)
    text = extract_text_from_content(response.content)

    assert len(text) > 20, "Descripción muy corta"
    print(f"\n✓ Test 6 pasado. Descripción de imagen remota:")
    print(text[:300])


# ************************************************************************
# Test 7: Text + 2 Images (remote) -> Text (Comparación de imágenes)
# ************************************************************************

def test_vision_multiple_images(api_key: str):
    """
    Test 7: Comparación de dos imágenes remotas usando modelo openai.
    Usa IMAGELOCATION_1 y IMAGELOCATION_2.
    """
    IMAGE_URL_1 = "https://i.ibb.co/CpZvrgGw/doki1.jpg"
    IMAGE_URL_2 = "https://i.ibb.co/gFrRgLQd/doki2.jpg"

    chat = ChatPollinations(
        api_key=api_key,
        request_defaults={"model": "openai", "max_tokens": 1400}
    )

    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "Compara estas dos imágenes. ¿En qué se parecen y en qué se diferencian?"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL_1}},
            {"type": "image_url", "image_url": {"url": IMAGE_URL_2}}
        ])
    ]

    response = chat.invoke(messages)
    text = extract_text_from_content(response.content)

    assert len(text) > 30, "Comparación muy corta"
    # Verificar que menciona ambas imágenes o hace comparación
    assert any(word in text.lower() for word in ["similar", "diferente", "ambas", "primera", "segunda", "parecen"]), \
        "No parece hacer comparación entre imágenes"

    print(f"\n✓ Test 7 pasado. Comparación de imágenes:")
    print(text[:400])


# ************************************************************************
# Test 8: Text + Image (remote) -> Text (OCR con gemini-fast)
# ************************************************************************

def test_vision_ocr_text_extraction(api_key: str):
    """
    Test 8: Extracción de texto de imagen (OCR) usando modelo gemini-fast.
    Usa IMAGELOCATION_3 que contiene una gráfica con texto.
    """
    IMAGE_URL = "https://i.ibb.co/v48NP6kg/grafica.jpg"

    chat = ChatPollinations(
        api_key=api_key,
        request_defaults={"model": "gemini-fast", "max_tokens": 1500}
    )

    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "Lee y extrae todo el texto visible en esta imagen. Si hay números o datos, inclúyelos también."},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}}
        ])
    ]

    response = chat.invoke(messages)
    text = extract_text_from_content(response.content)

    assert len(text) > 10, "Texto extraído muy corto"
    # Verificar que extrajo algún contenido textual/numérico
    has_alphanumeric = any(c.isalnum() for c in text)
    assert has_alphanumeric, "No se extrajo contenido alfanumérico de la imagen"

    print(f"\n✓ Test 8 pasado. Texto extraído de imagen (OCR):")
    print(text[:500])


# ************************************************************************
# Test 9: Thinking mode
# ************************************************************************

def test_thinking_blocks_with_empty_content(api_key: str):
    """
    Test específico para modelos con thinking habilitado.
    Verifica que content="" + content_blocks se maneja correctamente.
    """
    chat = ChatPollinations(
        api_key=api_key,
        request_defaults={
            "model": "deepseek",  # o "openai" con reasoning
            "thinking": {"type": "enabled", "budget_tokens": 1000}
        }
    )

    messages = [
        HumanMessage(content="Resuelve: Si tengo 3 manzanas y compro el doble, ¿cuántas tengo?")
    ]

    response = chat.invoke(messages)

    # Verificar que hay content_blocks
    content_blocks = response.additional_kwargs.get("content_blocks")

    if content_blocks:
        # Verificar estructura con TypedDict
        has_thinking = any(
            cast(ContentBlock, b).get("type") == "thinking"
            for b in content_blocks if isinstance(b, dict)
        )
        has_text = any(
            cast(ContentBlock, b).get("type") == "text"
            for b in content_blocks if isinstance(b, dict)
        )

        assert has_thinking or has_text, "No se detectaron bloques thinking o text"

        # Si content es string, verificar que extrajo solo el texto (no thinking)
        if isinstance(response.content, str):
            # No debe incluir el contenido de thinking
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_text = block.get("thinking", "")
                    assert thinking_text not in response.content, \
                        "Thinking block se filtró a content (debería estar solo en additional_kwargs)"

        # Si content es lista, debe ser content_blocks completo
        elif isinstance(response.content, list):
            assert response.content == content_blocks

    print(f"\n✓ Test thinking: content type={type(response.content)}")
    if content_blocks:
        print(f"  content_blocks: {len(content_blocks)} bloques")


# ************************************************************************
# Test 10: Streaming mode
# ************************************************************************

def test_streaming_empty_content_with_blocks(api_key: str):
    """
    Verifica que en streaming, deltas con content="" + content_blocks
    se manejan correctamente y se preservan en additional_kwargs.
    """
    chat = ChatPollinations(
        api_key=api_key,
        preserve_multimodal_deltas=True,
        request_defaults={"model": "openai-fast"}
    )

    messages = [HumanMessage(content="Di hola")]

    chunks_with_blocks = []
    all_chunks = []

    # En streaming, chat.stream() retorna AIMessageChunk directamente
    for chunk in chat.stream(messages):
        all_chunks.append(chunk)
        # Acceso directo a additional_kwargs (chunk es AIMessageChunk)
        if "content_blocks" in chunk.additional_kwargs:
            chunks_with_blocks.append(chunk)

    print(f"\n✓ Streaming test: {len(all_chunks)} chunks totales, {len(chunks_with_blocks)} con content_blocks")

    # Si hubo chunks con content_blocks, verificar estructura
    if chunks_with_blocks:
        for i, chunk in enumerate(chunks_with_blocks[:3]):  # Primeros 3
            blocks = chunk.additional_kwargs["content_blocks"]
            print(f"  Chunk {i}: {len(blocks)} bloques")

            # Verificar estructura usando TypedDict
            for block in blocks:
                if isinstance(block, dict):
                    block_typed = cast(ContentBlock, block)
                    print(f"    - type: {block_typed.get('type')}")
    else:
        # Es válido que no haya content_blocks en respuestas simples
        print("  (No se detectaron content_blocks - esto es normal para respuestas simples)")

        # Verificar que al menos hay contenido textual
        text_content = "".join([chunk.content for chunk in all_chunks if isinstance(chunk.content, str)])
        assert len(text_content) > 0, "No se recibió contenido en el streaming"
        print(f"  Contenido textual total: {len(text_content)} chars")


# ************************************************************************
# Test 11: Verificar manejo de content_blocks con TypedDicts
# ************************************************************************

def test_content_blocks_handling(api_key: str):
    """
    Test bonus: Verifica que content_blocks se maneja correctamente
    usando los TypedDicts definidos en _openai_compat.py.
    """
    chat = ChatPollinations(
        api_key=api_key,
        request_defaults={"model": "openai-fast"}
    )

    messages = [
        HumanMessage(content="Dame una respuesta corta: ¿qué es Python?")
    ]

    response = chat.invoke(messages)

    # Verificar que podemos acceder a content_blocks si existen
    content_blocks = response.additional_kwargs.get("content_blocks")

    if content_blocks and isinstance(content_blocks, list):
        print(f"\n✓ Content_blocks detectado ({len(content_blocks)} bloques)")

        for i, block in enumerate(content_blocks):
            if isinstance(block, dict):
                block_typed = cast(ContentBlock, block)
                block_type = block_typed.get("type")
                print(f"  Bloque {i}: type='{block_type}'")

                if block_type == "text":
                    text_block = cast(ContentBlockText, block_typed)
                    text_content = text_block.get("text", "")
                    print(f"    Text content: {text_content[:50]}...")
    else:
        # Si no hay content_blocks, verificar que content es string
        assert isinstance(response.content, str) or isinstance(response.content, list)
        print("\n✓ Respuesta sin content_blocks (solo content directo)")
