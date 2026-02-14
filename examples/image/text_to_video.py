import dotenv

from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

gen = ImagePollinations()
video_bytes = gen.generate(
    "Un cambur con pies, ojos, nariz y boca, bailando pop music sobre una mesa en un dormitorio lleno de juguetes, cinematográfico estilo caricatura de Pixar Studios",
    params={
        "model": "seedance",
        "duration": 6,
        "aspectRatio": "16:9",
        "seed": 15,
        "safe": True,
    },
)

with open("cambur.mp4", "wb") as f:
    f.write(video_bytes)