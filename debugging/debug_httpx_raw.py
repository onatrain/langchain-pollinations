import os
import json
import httpx
import dotenv

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
    print("text:", r.text)          # ya viene descomprimido por httpx
    print("json:", json.dumps(r.json(), ensure_ascii=False, indent=2))

