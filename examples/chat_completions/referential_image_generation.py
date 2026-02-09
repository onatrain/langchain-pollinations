import dotenv
from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

img = ImagePollinations()
data = img.generate(
    "Convierte esta foto en estilo ghibli, manteniendo la pose",
    params={
        "model": "klein",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Ovejera_Alem%C3%A1n.jpg/500px-Ovejera_Alem%C3%A1n.jpg",
        "width": 1024,
        "height": 1024,
    },
)

with open("anime.png", "wb") as f:
    f.write(data)