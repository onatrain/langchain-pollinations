import dotenv
from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

img = ImagePollinations()
data = img.generate(
    "Logo minimalista de un perro Yorkshire de pelaje color gris algo oscuro, fondo transparente",
    params={
        "model": "gptimage",
        "width": 256,
        "height": 256,
        "transparent": True,
        "quality": "high",
    },
)

with open("logo.png", "wb") as f:
    f.write(data)