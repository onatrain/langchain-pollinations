import os

import requests
import base64
import json
from pathlib import Path

def analyze_audio_with_pollinations_mp3(mp3_path: str, api_key: str, question: str):
    """
    Envía un archivo MP3 codificado en base64 directamente en el campo `content`
    del mensaje de role 'user' al endpoint /v1/chat/completions de Pollinations,
    usando el modelo `openai-audio`. No usa attachments: el audio se incluye
    como un string (data URI) dentro de la lista de content.

    Args:
        mp3_path: Ruta al archivo MP3 local
        api_key: el API Key secreta de enter.pollinations.ai (variable de entorno)
        question: Pregunta sobre el contenido del audio (p. ej. "¿De qué trata este audio?")
    """

    # 1. Leer y codificar el MP3 en base64
    try:
        with open(mp3_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {mp3_path}")
        return

    # 2. Construir data URI para MP3
    data_uri = f"data:audio/mpeg;base64,{audio_b64}"

    # 3. Endpoint y headers
    base_url = "https://gen.pollinations.ai"
    endpoint = f"{base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 4. Construir payload: modelo openai-audio, modalidades text (transcripción->respuesta)
    #    Mensaje user contiene una lista con el texto de la pregunta y un objeto audio
    #    representado como un simple dict con tipo 'audio_url' cuyo url es la data URI.
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": data_uri,
                    "format": "mp3"
                }
            }
        ]
    }
    payload = {
        "model": "openai-audio",
        "modalities": ["text"],          # solicitamos respuesta en texto (transcripción + QA)
        "messages": [
            {
                "role": "system",
                "content": "Eres un asistente que transcribe audio y responde preguntas sobre su contenido."
            },
            user_message
        ],
        "temperature": 0.2,
        "max_tokens": 800,
        "stream": False
    }

    # 5. Enviar petición POST
    try:
        print("Enviando MP3 codificado a Pollinations (modelo: openai-audio)...")
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error {resp.status_code}: {http_err}")
        print(resp.text)
        return
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Error en la petición: {req_err}")
        return

    # 6. Procesar respuesta
    try:
        result = resp.json()
    except ValueError:
        print("❌ Respuesta no es JSON:")
        print(resp.text)
        return

    if "choices" in result and len(result["choices"]) > 0:
        message = result["choices"][0]["message"]["content"]
        print("\n🔊 Transcripción / Respuesta:")
        print("-" * 60)
        print(message)
        print("-" * 60)
        if "usage" in result:
            usage = result["usage"]
            print("\n📊 Uso de tokens:")
            print(f"  Prompt: {usage.get('prompt_tokens','N/A')}")
            print(f"  Completion: {usage.get('completion_tokens','N/A')}")
            print(f"  Total: {usage.get('total_tokens','N/A')}")
    else:
        print("Respuesta inesperada:", json.dumps(result, indent=2))


if __name__ == "__main__":
    # Configuración
    import dotenv

    dotenv.load_dotenv()

    API_KEY = os.getenv("POLLINATIONS_API_KEY")
    AUDIO_FILE = "marycorto.mp3"     # Ruta al audio
    QUESTION = "Transcribe el contenido de este audio y menciona el idioma hablado."

    # Verificar que existe el archivo
    if not Path(AUDIO_FILE).exists():
        print(f"⚠️  Crea un archivo llamado '{AUDIO_FILE}' en este directorio o actualiza la ruta")
    else:
        analyze_audio_with_pollinations_mp3(AUDIO_FILE, API_KEY, QUESTION)
