import dotenv
from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

img = ImagePollinations()
data = img.generate(
    "Una perra German Shepherd corriendo en un campo cubierto de grama corta. Estilo fotorrealista, alta calidad",
    params={
        "model": "klein-large",
        "width": 1024,
        "height": 1024,
        "seed": 42,
        "enhance": True,
        "safe": True,
        "quality": "high",
        "negative_prompt": "blurry, worst quality",
    },
)

with open("sasha.png", "wb") as f:
    f.write(data)