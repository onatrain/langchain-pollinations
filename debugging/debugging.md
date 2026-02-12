# Guía rápida de debugging (langchain-pollinations)

Este documento reúne **los comandos y programas** que se pueden ejecutar para obtener evidencia (requests/responses) y diagnosticar problemas.

> Requisitos:
> - Tener `POLLINATIONS_API_KEY` disponible (en `.env` o exportada en el entorno).
> - Estar en la **raíz** del proyecto.
> - Ejecutar con `uv run ...` (o equivalente en tu entorno).

---

## 0) Verificación de entorno

### 0.1 Comprobar Python y versiones
Sirve para confirmar versiones exactas de runtime y dependencias (útil para reproducibilidad).

```bash
uv run python -V
uv run python -c "import httpx,langchain_core; print(httpx.__version__, langchain_core.__version__)"
```

### 0.2 Comprobar que existe la API key
Sirve para validar que la variable de entorno está cargada.

```bash
uv run python -c "import os, dotenv; dotenv.load_dotenv(); print(bool(os.getenv('POLLINATIONS_API_KEY')))"
```

---

## 1) Debug “raw”

### 1.1 Obtener respuesta raw con `curl` (no-stream)
Sirve para ver **headers** (cache, encoding) y el **JSON completo** devuelto por el endpoint.

```bash
curl -sS --compressed -D - \
  -H "Authorization: Bearer $POLLINATIONS_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Cache-Control: no-cache" \
  -d '{"messages":[{"role":"user","content":"Responde solo con la palabra: OK"}],"model":"openai","temperature":0.2,"max_tokens":64,"stream":false}' \
  https://gen.pollinations.ai/v1/chat/completions
```

Qué revisar en la salida:
- `content-type` (debe ser JSON).
- `x-cache`, `x-cache-key`, `x-request-id` (para correlación y cache).
- En el body: `choices[0].message.content` y/o `choices[0].message.content_blocks`.

### 1.2 Obtener SSE raw con `curl` (streaming)
Sirve para ver las **líneas `data:`** reales del stream.

```bash
curl -sS -N --compressed \
  -H "Authorization: Bearer $POLLINATIONS_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Cache-Control: no-cache" \
  -d '{"messages":[{"role":"user","content":"Di: hola"}],"model":"openai","temperature":0.2,"max_tokens":64,"stream":true}' \
  https://gen.pollinations.ai/v1/chat/completions
```

Qué revisar:
- Si llegan eventos `data: {...}` con `choices[0].delta.content`.
- Si aparece `data: [DONE]`.
- Si los chunks traen contenido vacío o campos alternos (`content_blocks`, etc.).

---

## 2) Debug raw con script `httpx` 

### 2.1 Script `debug_httpx_raw.py`
Sirve para ver el response tal cual lo da el endpoint, con `httpx` (incluye gzip).

Crear el archivo `debug_httpx_raw.py` en la raíz:

```python
import os
import dotenv
import json
import httpx

dotenv.load_dotenv()

url = "https://gen.pollinations.ai/v1/chat/completions"
api_key = os.environ["POLLINATIONS_API_KEY"]

payload = {
    "messages": [{"role": "user", "content": "Responde solo con la palabra: OK"}],
    "model": "openai",
    "temperature": 0.2,
    "stream": False,
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Cache-Control": "no-cache",
}

with httpx.Client(timeout=120.0) as client:
    r = client.post(url, headers=headers, json=payload)
    print("status:", r.status_code)
    print("headers:", dict(r.headers))
    print("text:", r.text)
    print("json:", json.dumps(r.json(), ensure_ascii=False, indent=2))
```

Ejecutar:

```bash
uv run python debug_httpx_raw.py
```

Qué revisar:
- `usage.completion_tokens_details.reasoning_tokens`
- `choices[0].message.content` vs `content_blocks`

---

## 3) Logging de requests/responses desde la librería (httpx)

### 3.1 Activar logging por variable de entorno
Sirve para registrar:
- request: método, URL, headers (redactados), body JSON.
- response: status, headers.
- response body SOLO en no-stream (en streaming se evita consumir el stream).

Ejecutar integración con debug activado:

```bash
POLLINATIONS_HTTP_DEBUG=1 uv run integration -m tests/ -q -s
```

Qué buscar en logs:
- Request body exacto enviado a `/v1/chat/completions`.
- `content-type` de la respuesta (JSON vs `text/event-stream`).
- Si la respuesta JSON viene comprimida (`content-encoding: gzip`).
- Correlación por `x-request-id`.

---

## 4) Logging “bajo nivel” del transporte (httpcore)

### 4.1 Activar logging de httpcore/httpx
Sirve para ver el flujo de red:
- conexión TCP/TLS,
- envío de headers/body,
- recepción de headers/body,
- y en streaming, si el body se corta prematuramente.

### 4.2 Configurar el logging en el programa o test que se quiere depurar

Agregar al inicio del programa o suite de tests:
```
if os.getenv("POLLINATIONS_HTTP_DEBUG", "").lower() in {"1", "true", "yes", "on"}:
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
    )

    # httpcore es el motor interno de httpx
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.DEBUG)
```

### 4.3 Ejecutar el programa o tests que se quieren depurar
Ejecutar tests:

```bash
POLLINATIONS_HTTP_DEBUG=1 uv run pytest -m integration -q -s
```
o el programa a depurar:

```bash
POLLINATIONS_HTTP_DEBUG=1 uv run program_to_debug.py
```

Qué revisar:
- `receive_response_headers.complete`
- `receive_response_body.started/complete`
- En streaming: si aparece `receive_response_body.failed exception=GeneratorExit()` (indica que el consumidor cortó el stream).

---

## 5) Debug del stream desde LangChain (qué objetos entrega)

### 5.1 Script para inspeccionar chunks (`debug_raw_stream.py`)
Sirve para ver:
- el tipo real de `chunk` (puede variar por versión),
- dónde está el contenido (`chunk.content` o `chunk.message.content`),
- y si hay textos vacíos desde el inicio.

Crear `debug_raw_stream.py`:

```python
import os, dotenv
from langchain_core.messages import HumanMessage
from langchain_pollinations.chat import ChatPollinations

dotenv.load_dotenv()

api_key = os.environ["POLLINATIONS_API_KEY"]

model = ChatPollinations(api_key=api_key, model="openai", temperature=0.2)

for i, chunk in enumerate(model.stream([HumanMessage(content="Di: hola")])):
    print("i=", i, "type=", type(chunk))
    print("repr=", repr(chunk))

    content = getattr(chunk, "content", None)
    if isinstance(content, str) and content:
        print("chunk.content:", repr(content))

    msg = getattr(chunk, "message", None)
    if msg is not None:
        msg_content = getattr(msg, "content", None)
        if isinstance(msg_content, str) and msg_content:
            print("chunk.message.content:", repr(msg_content))

    if i >= 30:
        break
```

Ejecutar:

```bash
uv run python debug_raw_stream.py
```

Qué revisar:
- si hay texto en algún chunk,
- si todos llegan con `content=""`,
- y si el stream finaliza sin nunca emitir tokens.

---

## 6) Flujo recomendado (orden)

1) `curl` no-stream para ver JSON/headers.
2) `curl` streaming para ver SSE raw.
3) `debug_httpx_raw.py` para confirmar respuesta sin la librería.
4) Activar logging de la librería.
5) Inspeccionar chunks con `debug_raw_stream.py` si el problema es solo streaming.
