# langchain-pollinations

**langchain-pollinations** provides LangChain-native wrappers for the [Pollinations.ai](https://enter.pollinations.ai) provider, designed to plug into the modern LangChain ecosystem (v1.2x) while staying strictly aligned with Pollinations.ai endpoints.

It includes:
- `ChatPollinations`: a chat model wrapper for the OpenAI-compatible endpoint `POST /v1/chat/completions`.
- `ImagePollinations`: an image/video generation wrapper for `GET /image/{prompt}` (returns bytes and can optionally return the full HTTP response for header-aware saving).

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

### 1) Generate an image (bytes) with instance defaults (ChatOpenAI-like)

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

### 2) Generate and save using `Content-Type` (via `generate_response()`)

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
    aspectRatio="16:9",
    audio=True,
)

resp = img.generate_response("a drone flying over a city at sunset, cinematic")
ext = ext_from_content_type(resp.headers.get("content-type", ""))
out_file = f"out{ext}"

with open(out_file, "wb") as f:
    f.write(resp.content)

print("OK:", out_file, "ctype:", resp.headers.get("content-type"), "bytes:", len(resp.content))
```

## Configuration notes

- `ChatPollinations` targets the OpenAI-compatible chat endpoint and supports tool calling and streaming patterns commonly used by LangChain agents.
- `ImagePollinations` targets `GET /image/{prompt}` and exposes most query parameters supported by the provider (model, size, seed, enhance, negative prompt, safe mode, quality/transparent for supported models, reference image URLs, video duration/aspect ratio/audio for supported video models).
- Both wrappers accept environment-based authentication (`POLLINATIONS_API_KEY`) for an ergonomic “just works” developer experience.

## Contributing

Issues and PRs are welcome—especially around edge-case compatibility with LangChain agent/tool flows and improved ergonomics for media saving and content-type handling.

## License

This work is shared to you under MIT License terms and conditions.
