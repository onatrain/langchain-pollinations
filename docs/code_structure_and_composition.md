# Estructura y composición del código (langchain-pollinations)

## 1) Mapa de módulos

La librería se organiza en el paquete `langchain_pollinations`, exponiendo una API pública concisa y delegando la lógica compleja a módulos internos de soporte.

Estructura de archivos y responsabilidades:
```text
langchain_pollinations/
  __init__.py              -> Exporta la API pública (Chat, Image, Account, Models, Errors).
  chat.py                  -> ChatPollinations (BaseChatModel), configuración (Pydantic) y binding de tools.
  image.py                 -> ImagePollinations (generación de imágenes/video) y ImagePromptParams.
  account.py               -> AccountInformation (perfil, balance, uso) y AccountUsageParams.
  models.py                -> ModelInformation (listado y filtrado de modelos).
  _auth.py                 -> AuthConfig (resolución de API Key).
  _client.py               -> PollinationsHttpClient (wrapper httpx sync/async, manejo de headers y logging).
  _errors.py               -> PollinationsAPIError y jerarquía de excepciones.
  _sse.py                  -> Parser minimalista de Server-Sent Events.
  _openai_compat.py        -> Normalización de mensajes, conversión de tools, tipos para audio/thinking y filtros de contenido.
```

## 2) API pública vs interna
La superficie pública se define en `__init__.py` mediante `__all__`:
- **Core**: `ChatPollinations`, `ImagePollinations`.
- **Información**: `AccountInformation`, `ModelInformation`.
- **Errores**: `PollinationsAPIError`.

Los módulos iniciados con guion bajo (`_client.py`, `_openai_compat.py`, etc.) son de uso interno exclusivo para la librería, encapsulando la complejidad de la comunicación HTTP y la serialización de datos.

## 3) Composición por responsabilidades

### Chat (`chat.py` + `_openai_compat.py`)
- **ChatPollinations**: Implementa `BaseChatModel`. Gestiona el ciclo de vida del request, configuración (`ChatPollinationsConfig`), y métodos estándar de LangChain (`invoke`, `stream`, `bind_tools`).
- **Compatibilidad**: `_openai_compat.py` actúa como puente de traducción.
    - **Input**: Convierte mensajes LangChain a diccionarios JSON compatibles con la API (`lc_messages_to_openai`), normalizando contenido multimodal (audio, imágenes).
    - **Output**: Define `TypedDicts` para estructuras complejas de respuesta (`AudioTranscript`, `ContentBlockThinking`, `ContentFilterResult`).
    - **Tools**: Transforma definiciones de herramientas (Pydantic, funciones, dicts) al esquema JSON esperado (`tool_to_openai_tool`).

### Imagen (`image.py`)
- **ImagePollinations**: Wrapper configurable para el endpoint `/image/{prompt}`.
- **Validación**: Utiliza `ImagePromptParams` (Pydantic) para validar y serializar parámetros de query string como `model`, `width`, `height`, `seed`, `enhance` y `aspect_ratio`.
- **Fluent Interface**: Método `with_params()` para crear nuevas instancias con configuración ajustada de forma inmutable.

### Cuenta y Modelos (`account.py`, `models.py`)
- **AccountInformation**: Clases de datos (`dataclass`) que exponen métodos para recuperar perfil, balance y reportes de uso. Utiliza `AccountUsageParams` para filtrar logs de consumo.
- **ModelInformation**: Utilidades para listar modelos disponibles, categorizándolos en texto e imagen y extrayendo identificadores normalizados.

### Transporte y Seguridad (`_client.py`, `_auth.py`)
- **PollinationsHttpClient**: Cliente HTTP robusto sobre `httpx`. Maneja:
    - Ciclo de vida de clientes síncronos y asíncronos.
    - Parsing automático de errores JSON a `PollinationsAPIError`.
    - Logging de depuración con redacción de credenciales (`Authorization`).
- **AuthConfig**: Centraliza la lógica de obtención de la API Key desde argumentos o variables de entorno.

## 4) Flujo de datos (Chat)

El flujo de una petición de chat atraviesa varias capas de transformación para soportar características avanzadas como multimodalidad y "thinking".

1.  **Entrada**: El usuario invoca `ChatPollinations.invoke(messages)`.
2.  **Normalización**: `lc_messages_to_openai` procesa la lista de mensajes.
    - `HumanMessage` con contenido multimodal se normaliza.
    - `ToolMessage` se simplifica para cumplir con el esquema estricto del proveedor.
    - `AIMessage` con `tool_calls` se convierte al formato de OpenAI.
3.  **Construcción del Payload**: Se combinan los mensajes procesados con la configuración del modelo (`model`, `temperature`, `tools`, etc.).
4.  **Transporte**: `PollinationsHttpClient` ejecuta `post_json` (o `stream_post_json`).
5.  **Procesamiento de Respuesta**:
    - **Síncrono**: Se parsea el JSON y `_message_content_from_message_dict` extrae el contenido, priorizando texto pero recuperando bloques de `thinking` o transcripciones de audio si están presentes.
    - **Streaming**: El iterador consume eventos SSE. `_delta_content_from_delta_dict` acumula fragmentos de texto o estructuras de `content_blocks` (para razonamiento incremental).
6.  **Salida**: Se retorna un `AIMessage` (o `AIMessageChunk`) que puede incluir `additional_kwargs` con metadatos de seguridad (`prompt_filter_results`) o estructuras de razonamiento (`thinking`).

## 5) Manejo de Tipos y Estructuras Auxiliares
La librería utiliza extensivamente `TypedDict` y `Pydantic` para garantizar la corrección de los datos:
- **Audio y Multimodalidad**: `_openai_compat.py` define `AudioTranscript` y lógica para inferir formatos de audio (`_infer_audio_format_from_mime`) y normalizar entradas `input_audio`.
- **Seguridad y Filtros**: Estructuras como `ContentFilterResult` y `PromptFilterResultItem` permiten mapear detalladamente las respuestas de moderación de contenido del API.
- **Reasoning**: Soporte explícito para bloques `thinking` y `redacted_thinking`, permitiendo a los modelos exponer sus cadenas de pensamiento antes de la respuesta final.