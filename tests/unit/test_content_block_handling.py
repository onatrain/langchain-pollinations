from langchain_pollinations.chat import (
    _message_content_from_message_dict,
    _delta_content_from_delta_dict
)


def test_message_content_empty_string_with_blocks():
    """
    Verifica que content="" no cortocircuite cuando hay content_blocks.
    Este es el caso edge que motiva el cambio en la lógica.
    """
    message = {
        "content": "",  # String vacío explícito
        "content_blocks": [
            {"type": "thinking", "thinking": "Analizando..."},
            {"type": "text", "text": "Respuesta final"}
        ]
    }

    result = _message_content_from_message_dict(message)

    # DEBE retornar content_blocks, NO el string vacío
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["type"] == "thinking"
    assert result[1]["type"] == "text"


def test_delta_content_empty_string_with_blocks():
    """
    Verifica lo mismo para deltas en streaming.
    """
    delta = {
        "content": "",
        "content_blocks": [{"type": "text", "text": "chunk"}]
    }

    result = _delta_content_from_delta_dict(delta)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["text"] == "chunk"


def test_message_content_none_with_blocks():
    """
    Verifica que content=None también permite buscar content_blocks.
    """
    message = {
        "content": None,
        "content_blocks": [{"type": "text", "text": "texto"}]
    }

    result = _message_content_from_message_dict(message)
    assert isinstance(result, list)


def test_message_content_list_multimodal():
    """
    Verifica que content como lista se retorna directamente.
    """
    message = {
        "content": [
            {"type": "text", "text": "parte1"},
            {"type": "thinking", "thinking": "pensando..."}
        ]
    }

    result = _message_content_from_message_dict(message)
    assert isinstance(result, list)
    assert result == message["content"]


def test_message_content_empty_list_with_blocks():
    """
    Caso edge: content=[] vacía, pero hay content_blocks.
    """
    message = {
        "content": [],
        "content_blocks": [{"type": "text", "text": "fallback"}]
    }

    result = _message_content_from_message_dict(message)
    # content=[] vacía NO debería retornarse, debe buscar blocks
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["text"] == "fallback"


def test_normal_string_content_still_works():
    """
    Regresión: content con texto normal debe seguir funcionando.
    """
    message = {"content": "Texto normal sin bloques"}
    result = _message_content_from_message_dict(message)
    assert result == "Texto normal sin bloques"


def test_content_priority_over_blocks_when_not_empty():
    """
    Si content tiene texto Y hay blocks, content debe tener prioridad.
    """
    message = {
        "content": "Respuesta directa",
        "content_blocks": [{"type": "text", "text": "Otra cosa"}]
    }
    result = _message_content_from_message_dict(message)
    assert result == "Respuesta directa"
