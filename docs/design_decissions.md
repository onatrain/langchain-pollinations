# Decisiones de diseño de la API (langchain-pollinations)

Este documento describe las decisiones de diseño que guían la **API pública** de la librería y cómo se espera que se use.

## 1) Interfaz pública mínima y clara

- La librería expone solo cuatro “entradas” principales como API pública: `ChatPollinations`, `ImagePollinations`, `AccountInformation` y `ModelInformation`.
- La intención es que el usuario pueda cubrir los 3 casos típicos (chat, imágenes, info de cuenta/modelos) sin conocer detalles internos (auth, http client, compat OpenAI, etc.).
- Las utilidades internas viven en módulos “privados” (prefijo `_...`) para permitir cambios sin romper a usuarios.

Ejemplo de imports recomendados:
```python
from langchain_pollinations import ChatPollinations, ImagePollinations
from langchain_pollinations import AccountInformation, ModelInformation
```

## 2) Integración con LangChain (chat)

- `ChatPollinations` hereda de `BaseChatModel` para integrarse con el ecosistema LangChain: `invoke()`, `stream()`, callbacks, y el modelo de mensajes `HumanMessage/SystemMessage/AIMessage`.
- La API de chat usa el endpoint OpenAI-compatible `/v1/chat/completions` como “contrato” de request/response, pero sin obligar al usuario a hablar “OpenAI JSON”; se aceptan mensajes LangChain y se convierten internamente.
- Se soporta “tool calling” vía `bind_tools()`, transformando herramientas a formato OpenAI `tools=[{type:"function", function:{...}}]`.

## 3) Configuración: estricta y validada (Pydantic)

- Los parámetros del request body de `/v1/chat/completions` se modelan con `ChatPollinationsConfig` (Pydantic) y se valida con `extra="forbid"` para detectar campos inválidos/typos temprano.
- `ChatPollinations.__init__` separa explícitamente:
  - kwargs que pertenecen al request (se validan contra `ChatPollinationsConfig`),
  - kwargs que pertenecen al `BaseChatModel` (callbacks, tags, metadata, etc.).
- Se ofrece `request_defaults` para configurar defaults “de sesión” sin repetir parámetros en cada llamada.

## 4) Autenticación simple y explícita

- La API key se obtiene desde:
  1) el argumento `api_key=...` si se pasa,
  2) o la variable de entorno `POLLINATIONS_API_KEY`.
- Si no hay key, se falla rápido con un `ValueError` con mensaje claro (evita “silent misconfig”).

## 5) Transporte HTTP: httpx, sync/async, errores consistentes

- Todo el transporte HTTP se centraliza en `PollinationsHttpClient` (interno), construido con:
  - `HttpConfig(base_url, timeout_s)` para parametrizar host y timeout,
  - `httpx.Client` y `httpx.AsyncClient` para soportar sync y async en paralelo.
- Los wrappers públicos reflejan esa decisión:
  - Chat: `_generate/_agenerate`, `_stream/_astream`
  - Imágenes: `generate/agenerate`
  - Modelos: `list_*/alist_*`
- Los errores HTTP no-2xx se elevan como `PollinationsAPIError(status_code, message, body)` para tener una excepción estable y fácil de inspeccionar.

## 6) Streaming (SSE): pragmático y compatible

- El streaming de chat usa `Accept: text/event-stream` y consume el stream iterando líneas `data: ...`, parseando JSON por evento.
- Se incluye un parser SSE mínimo (`iter_sse_events_from_text`) pensado para pruebas unitarias o parsing offline (no para “streaming real” de red).
- El diseño privilegia:
  - entregar chunks compatibles con LangChain (`AIMessageChunk`/`ChatGenerationChunk`),
  - y mantener el parsing SSE lo suficientemente simple para debug y mantenimiento.

## 7) Imagen, modelos y cuenta: wrappers delgados

- `ImagePollinations` es un wrapper directo de `GET /image/{prompt}` y retorna `bytes` (no intenta adivinar formato final).
- `ModelInformation` encapsula endpoints de listado (`/v1/models`, `/text/models`, `/image/models`) para que el usuario descubra modelos sin hardcodear.
- `AccountInformation` encapsula endpoints `/account/*` y tipa parámetros de query con Pydantic (`format`, `limit`, etc.) con validación estricta.
