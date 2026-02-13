<div align="center">
    <table>
        <tr>
            <td width="128px">
                <img src="assets/doki.png" alt="langchain-pollinations" width="128px"/>
            </td>
            <td align="left">
                <h1>langchain-pollinations</h1>
                <p><strong>A LangChain compatible provider library for Pollinations.ai</strong></p>
            </td>
        </tr>
    </table>

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/onatrain/langchain-pollinations)
[![Coverage](https://img.shields.io/badge/coverage-96%25-9C27B0)](https://github.com/onatrain/langchain-pollinations)
[![Status](https://img.shields.io/badge/status-beta-orange)](https://github.com/onatrain/langchain-pollinations)
[![License](https://img.shields.io/github/license/onatrain/langchain-pollinations?label=license&color=97CA00)](LICENSE.md)
[![Python Versions](https://img.shields.io/badge/python-3.11+-3776AB?logo=python)](https://github.com/onatrain/langchain-pollinations)
<br>
[![LangChain](https://img.shields.io/badge/langchain-1d3d3c?logo=langchain)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/langgraph-053d5b?logo=langgraph)](https://www.langchain.com/langgraph)

</div>

--- 

**langchain-pollinations** provides LangChain-native wrappers for the [Pollinations.ai](https://enter.pollinations.ai) provider, designed to plug into the modern LangChain ecosystem (v1.2x) while staying strictly aligned with Pollinations.ai endpoints.

It includes:
- `ChatPollinations`: a chat model wrapper for the OpenAI-compatible endpoint `POST /v1/chat/completions`.
- `ImagePollinations`: an image/video generation wrapper for `GET /image/{prompt}`.
- `ModelInformation`: a utility to list available models for text, image, and OpenAI-compatible endpoints.
- `AccountInformation`: a client to check profile details, balance, and API usage statistics.

## Why this project

Pollinations.ai offers a unified gateway for text, vision, tools, and media generation. This project makes it easy to use that gateway with LangChain patterns (invoke/stream, tool calling, and message formats) while keeping the surface area small and predictable.

## Installation

```bash
pip install langchain-pollinations
```

## Authentication

Copy the *.env.example* file to **.env** file and then open and edit this new file and change the line:

```
POLLINATIONS_API_KEY=YOUR_SECRET_KEY
```

with your key **"sk_...your_key..."** obtained from [Pollinations](https://enter.pollinations.ai)

Or pass `api_key=...` parameter explicitly when constructing clients.

## ChatPollinations examples

### 1) Basic chat completion

```python
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

llm = ChatPollinations(model="openai")

res = llm.invoke([
    HumanMessage(content="Write a short haiku about distributed systems.")
])

print(res.content)
```

### 2) Vision question (image URL input)

```python
import dotenv
from langchain_pollinations import ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

IMAGE_URL = "https://example.com/image.jpg"  # replace with a real URL

llm = ChatPollinations(model="openai")

msg = HumanMessage(content=[
    {"type": "text", "text": "Describe the image in one sentence."},
    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
])

res = llm.invoke([msg])
print(res.content)
```

## ImagePollinations examples

### 1) Generate an image (bytes) with instance defaults

```python
import dotenv
from langchain_pollinations.image import ImagePollinations

dotenv.load_dotenv()

img = ImagePollinations(
    model="klein",
    width=1024,
    height=1024,
    seed=42,
)

data = img.generate("a cute dog, studio lighting")

with open("cute_dog.jpg", "wb") as f:
    f.write(data)

print("OK: cute_dog.jpg", "bytes:", len(data))
```

### 2) Generate and save using `Content-Type`

```python
import dotenv
from langchain_pollinations.image import ImagePollinations

dotenv.load_dotenv()

def ext_from_content_type(ct: str) -> str:
    ct = (ct or "").split(";").strip().lower()
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
    }.get(ct, ".bin")

img = ImagePollinations(
    model="veo",
    duration=6,
    aspect_ratio="16:9",
    audio=True,
)

resp = img.generate_response("a drone flying over a city at sunset, cinematic")
ext = ext_from_content_type(resp.headers.get("content-type", ""))
out_file = f"out{ext}"

with open(out_file, "wb") as f:
    f.write(resp.content)

print("OK:", out_file, "ctype:", resp.headers.get("content-type"), "bytes:", len(resp.content))
```

## ModelInformation examples

### 1) List available text models

```python
from langchain_pollinations.models import ModelInformation

models = ModelInformation()
text_models = models.list_text_models()

for model in text_models:
    print(f"Model ID: {model['id']} - Name: {model['name']}")
```

### 2) Get OpenAI-compatible V1 models (Async)

```python
import asyncio
from langchain_pollinations.models import ModelInformation

async def show_v1_models():
    info = ModelInformation()
    v1_list = await info.alist_v1_models()
    print("Available V1 Models:", v1_list.keys())

asyncio.run(show_v1_models())
```

## AccountInformation examples

### 1) Check current balance and profile

```python
from langchain_pollinations.account import AccountInformation

account = AccountInformation()
profile = account.get_profile()
balance = account.get_balance()

print(f"User: {profile['username']} | Balance: {balance['credits']} credits")
```

### 2) Fetch daily usage statistics

```python
from langchain_pollinations.account import AccountInformation, AccountUsageDailyParams

account = AccountInformation()
params = AccountUsageDailyParams(format="json")
usage = account.get_usage_daily(params=params)

print("Daily Usage Summary:", usage)
```

## General features

- `ChatPollinations` targets the OpenAI-compatible chat endpoint and supports tool calling and streaming patterns.
- `ImagePollinations` targets `GET /image/{prompt}` and exposes query parameters like seed, enhance, and video duration.
- `ModelInformation` provides access to `/text/models`, `/image/models`, and `/v1/models` endpoints.
- `AccountInformation` allows monitoring of `/account/profile`, `/account/balance`, and `/account/usage`.
- All wrappers accept environment-based authentication (`POLLINATIONS_API_KEY`) or explicit `api_key` parameters.

## Contributing

Issues and PRs are welcome—especially around edge-case compatibility with LangChain agent/tool flows and improved ergonomics for media saving.

## License

This work is shared to you under MIT License terms and conditions.