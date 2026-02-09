import dotenv
from langchain_pollinations.image import ImagePollinations

dotenv.load_dotenv()

prompt = "a funky painted, vivid coloured Volkswagen Beetle in middle of a desert at day, photorealistic image"

params = {
    "model": "klein-large",   # prueba: flux, zimage, turbo, kontext, seedream, gptimage, etc.
    "width": 1024,
    "height": 576,
    "seed": -1,
    "enhance": True,
    "negative_prompt": "worst quality, blurry, lowres, artifacts",
    "safe": False,
    "quality": "hd",
}

out_file = "custom.jpg"
img = ImagePollinations()
data = img.generate(prompt, params=params)

with open(out_file, "wb") as f:
    f.write(data)

print("OK:", out_file)