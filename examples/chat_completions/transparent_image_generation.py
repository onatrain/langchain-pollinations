import dotenv
from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

img = ImagePollinations()
data = img.generate(
    "Logo minimalista de un perro Yorkshire de pelaje color gris algo oscuro, fondo transparente",
    params={
        "model": "klein-large",
        "width": 1024,
        "height": 1024,
        "transparent": True,
        "quality": "high",
    },
)

with open("logo.png", "wb") as f:
    f.write(data)