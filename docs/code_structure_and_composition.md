# Estructura y composición del código (langchain-pollinations)

## 1) Mapa de módulos

La librería está organizada como un paquete `langchain_pollinations` con una API pública pequeña y varios módulos 
internos “de soporte” (auth, HTTP, compatibilidad OpenAI). 

Los componentes principales se reparten entre wrappers de endpoints (chat, image, account, models) y utilidades 
transversales (errores, SSE, conversión de mensajes). 

Estructura conceptual: 
```text
langchain_pollinations/
  __init__.py              -> re-export de API pública
  chat.py                  -> ChatPollinations (LangChain ChatModel) + tipos OpenAI
  image.py                 -> ImagePollinations (wrapper endpoint imagen)
  account.py               -> AccountInformation (endpoints /account/*)
  models.py                -> ModelInformation (endpoints de modelos)
  _sse.py                   -> parser SSE mínimo (útil en tests/offline)
  _auth.py                 -> AuthConfig (API key desde env o parámetro)
  _client.py               -> PollinationsHttpClient (httpx sync/async)
  _errors.py               -> excepciones propias (PollinationsAPIError)
  _openai_compat.py        -> conversión LC messages/tools -> OpenAI JSON
```

## 2) API pública vs interna
La API pública se expone desde `__init__.py` mediante un `__all__` explícito, lo cual ayuda a mantener una superficie 
estable de imports para usuarios.

Las piezas internas (`AuthConfig`, `PollinationsHttpClient`, utilidades de compatibilidad OpenAI) se consumen desde 
los wrappers, pero no se promueven como “entry points” del usuario final. 

## 3) Composición por responsabilidades
**Chat**: `chat.py` concentra (a) el modelo `ChatPollinations` compatible con LangChain y (b) los tipos/validación 
Pydantic del request body de `/v1/chat/completions` (por ejemplo `ChatPollinationsConfig`, `ToolDef`, `ResponseFormat`, 
etc.).

**HTTP/Auth**: `_auth.py` resuelve la API key (env o argumento) y `_client.py` encapsula `httpx.Client`/`httpx.AsyncClient` 
con métodos `get/post_json` (sync/async) y normaliza errores HTTP a `PollinationsAPIError`.  

**Otros endpoints**: `image.py`, `models.py` y `account.py` son wrappers delgados (dataclasses) que inicializan un 
`PollinationsHttpClient` y exponen métodos directos hacia endpoints específicos. 

## 4) Flujo de datos (chat)
Entrada: el usuario llama `invoke()`/`stream()` sobre `ChatPollinations` pasando `list[BaseMessage]` (mensajes 
LangChain). 

Transformación: `lc_messages_to_openai()` convierte esos mensajes a la forma `{"role": ..., "content": ...}` y 
`_build_payload()` arma el payload final aplicando defaults (`request_defaults`) y validando con Pydantic.

Salida: `_generate()` hace POST JSON y `_parse_chat_result()` convierte `choices[0].message` a `AIMessage`, mientras 
que `_stream()` abre un stream SSE y va emitiendo `AIMessageChunk`/`ChatGenerationChunk` al parsear cada `data: ...`.

Diagrama simplificado del call flow:
```text
User -> ChatPollinations.invoke(messages)
    -> lc_messages_to_openai(messages)
    -> _build_payload(...)
    -> PollinationsHttpClient.post_json("/v1/chat/completions", payload)
    -> resp.json()
    -> _parse_chat_result(data)
    -> AIMessage (LangChain)
```

## 5) Piezas auxiliares (SSE, tools, errores)
`sse.py` incluye un parser SSE mínimo basado en separar por doble salto de línea y extraer líneas `data:`, orientado a 
parsing offline o pruebas, no a streaming “en vivo” de red. 

`_openai_compat.py` también se encarga de mapear herramientas a `tools` en formato OpenAI (`tool_to_openai_tool`) y de 
trasladar campos relevantes como `tool_call_id` en `ToolMessage`. 

`_errors.py` define una jerarquía propia (`PollinationsError` y `PollinationsAPIError`) para que los consumidores 
puedan capturar errores de forma consistente sin depender de excepciones de httpx directamente. 
