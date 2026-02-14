import dotenv
from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

img = ImagePollinations()
data = img.generate(
    "Convierte esta foto en estilo pixar, manteniendo la pose",
    params={
        "model": "klein",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Ovejera_Alem%C3%A1n.jpg/500px-Ovejera_Alem%C3%A1n.jpg",
        "width": 512,
        "height": 512,
    },
)

with open("pixar.png", "wb") as f:
    f.write(data)