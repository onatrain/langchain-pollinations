import requests
import base64
import json
from pathlib import Path

def analyze_image_with_pollinations(image_path: str, api_key: str, model_name: str):
    """
    Envía una imagen JPEG codificada en base64 al endpoint /v1/chat/completions
    de Pollinations AI usando el modelo openai-large para descripción de contenido.

    Args:
        image_path: Ruta al archivo JPEG local
        api_key: Tu API Key secreta de enter.pollinations.ai
    """

    # 1. Leer y codificar la imagen en base64
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {image_path}")
        return

    # 2. Construir el data URI formato OpenAI-compatible
    data_uri = f"data:image/jpeg;base64,{encoded_image}"

    # 3. Configurar el endpoint y headers
    base_url = "https://gen.pollinations.ai"
    endpoint = f"{base_url}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 4. Construir el payload multimodal (formato OpenAI-compatible)
    payload = {
        "model": model_name,  # Modelo especificado
        "messages": [
            {
                "role": "system",
                "content": "Eres un asistente experto en análisis de imágenes. Describe detalladamente lo que ves."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "¿Qué hay en esta imagen? Describe los detalles principales."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500,  # Ajustar según la longitud de respuesta deseada
        "stream": False     # Cambiar a True para streaming
    }

    # 5. Realizar la petición POST
    try:
        print(f"Enviando imagen a Pollinations AI (modelo: {model_name})...")
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        # 6. Procesar respuesta
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            message_content = result["choices"][0]["message"]["content"]
            print("\n📝 Descripción de la imagen:")
            print("-" * 50)
            print(message_content)
            print("-" * 50)

            # Mostrar uso de tokens si está disponible
            if "usage" in result:
                usage = result["usage"]
                print(f"\n📊 Uso de tokens:")
                print(f"   Prompt: {usage.get('prompt_tokens', 'N/A')}")
                print(f"   Completion: {usage.get('completion_tokens', 'N/A')}")
                print(f"   Total: {usage.get('total_tokens', 'N/A')}")
        else:
            print("Respuesta inesperada:", json.dumps(result, indent=2))

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 401:
            print("❌ Error de autenticación: Verifica tu API Key")
        elif response.status_code == 429:
            print("❌ Rate limit excedido: Has alcanzado el límite de peticiones")
        else:
            print(f"❌ Error HTTP {response.status_code}: {http_err}")
            print(f"Detalle: {response.text}")

    except requests.exceptions.RequestException as req_err:
        print(f"❌ Error en la petición: {req_err}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


if __name__ == "__main__":
    # Configuración
    import dotenv, os

    # dotenv.load_dotenv()

    # API_KEY = os.getenv("POLLINATIONS_API_KEY")
    API_KEY = "plomo al hampa"
    IMAGE_FILE = "groovy.jpg"     # Reemplaza con la ruta a tu imagen JPEG
    MODEL = "gemini-fast"

    # Verificar que existe el archivo
    if not Path(IMAGE_FILE).exists():
        print(f"⚠️  Crea un archivo llamado '{IMAGE_FILE}' en este directorio o actualiza la ruta")
    else:
        analyze_image_with_pollinations(IMAGE_FILE, API_KEY, MODEL)