# Decisiones de Diseño API langchain-pollinations

Este documento detalla las decisiones de arquitectura y diseño que fundamentan la **API pública** y la estructura interna de la librería, asegurando consistencia con el ecosistema LangChain y la API de Pollinations.ai.

## 1) Interfaz pública minimalista

- **Superficie Reducida**: La librería expone únicamente cuatro puntos de entrada principales en `__init__.py`: `ChatPollinations`, `ImagePollinations`, `AccountInformation` y `ModelInformation`.
- **Ocultamiento de Complejidad**: Módulos internos como `_client`, `_auth`, `_sse` y `_openai_compat` manejan la complejidad del transporte, autenticación y normalización de datos, permaneciendo invisibles para el usuario final.
- **Jerarquía de Excepciones**: Se expone una única clase de error principal, `PollinationsAPIError`, que encapsula detalles HTTP y del backend (códigos de error, request ID, timestamps) para facilitar la depuración sin exponer excepciones de librerías subyacentes como `httpx`.

## 2) Integración Profunda con LangChain (Chat)

- **Compatibilidad Nativa**: `ChatPollinations` hereda de `BaseChatModel`, soportando flujos estándar (`invoke`, `stream`, `batch`) y el sistema de mensajes de LangChain (`System`, `Human`, `AI`, `Tool`).
- **Normalización Multimodal**:
    - El modelo acepta contenido multimodal (imágenes, audio) en los mensajes.
    - Internamente, `_openai_compat` normaliza estos inputs (base64, URLs) al formato estricto que espera la API, gestionando automáticamente tipos MIME y formatos de audio (`mp3`, `wav`, etc.).
- **Soporte de "Reasoning"**: Se diseñó soporte explícito para bloques de pensamiento (`thinking` y `redacted_thinking`). Estos se preservan o filtran según la configuración, permitiendo acceder a la cadena de razonamiento de modelos avanzados.

## 3) Configuración Estricta y Tipada (Pydantic)

- **Validación en Tiempo de Ejecución**: Todos los parámetros de configuración (tanto para chat como para generación de imágenes) se definen mediante modelos Pydantic (`ChatPollinationsConfig`, `ImagePromptParams`).
- **Política de "Extra Forbid"**: Se configura `extra="forbid"` en los modelos de configuración para rechazar inmediatamente parámetros desconocidos o mal escritos, evitando errores silenciosos en las peticiones a la API.
- **Separación de Responsabilidades**: En el constructor de `ChatPollinations`, se distinguen explícitamente los parámetros de configuración de la API (pasados a `request_defaults`) de los parámetros de configuración de LangChain (callbacks, tags).

## 4) Generación de Imágenes: Fluent Interface

- **Inmutabilidad y Encadenamiento**: `ImagePollinations` implementa un patrón "Fluent Interface" mediante el método `with_params()`. Esto permite crear nuevas instancias pre-configuradas (ej. un generador específico para "pixel art") sin mutar el objeto original.
- **Abstracción de Query Params**: La clase `ImagePromptParams` mapea y valida los numerosos parámetros de URL (`seed`, `width`, `height`, `model`, `enhance`) antes de construir la query string, asegurando que solo se envíen valores válidos.

## 5) Transporte HTTP Robusto y Centralizado

- **Cliente Unificado**: `PollinationsHttpClient` centraliza toda la lógica de red, manejando tanto peticiones síncronas como asíncronas sobre `httpx`.
- **Manejo de Errores Estructurado**: El cliente parsea automáticamente las respuestas de error JSON del backend, poblando `PollinationsAPIError` con detalles estructurados (`details`, `cause`, `requestId`) en lugar de solo texto plano.
- **Seguridad en Logging**: El sistema de logging interno redacta automáticamente headers sensibles (`Authorization`) para evitar fugas de credenciales en los logs de depuración.

## 6) Streaming y Eventos (SSE)

- **Parsing Resiliente**: Se implementó un parser SSE ligero (`_sse.py`) que maneja la transmisión de eventos.
- **Reconstrucción de Deltas**: En el modo streaming, la librería es capaz de reconstruir objetos complejos (como `content_blocks` para thinking o tool calls fragmentados) a partir de los deltas parciales recibidos.
- **Preservación de Datos**: Se decidió incluir opciones como `preserve_multimodal_deltas` para no perder información rica (audio, imágenes) durante la transmisión por chunks.

## 7) Tool Calling Adaptativo

- **Conversión Bidireccional**: La librería no solo convierte definiciones de herramientas de LangChain al formato OpenAI, sino que también implementa lógica específica (`_lc_tool_call_to_openai_tool_call`) para adaptar las llamadas a las particularidades del proveedor (ej. evitar el campo `name` en mensajes de respuesta de herramientas).
- **Inferencia de Esquemas**: Se soporta la inferencia automática de esquemas JSON para herramientas definidas como funciones Python, modelos Pydantic o diccionarios `TypedDict`.