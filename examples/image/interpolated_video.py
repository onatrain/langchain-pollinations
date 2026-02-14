import dotenv

from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

gen = ImagePollinations()
video_bytes = gen.generate(
    "Perro sentado que mira a la cámara y después de 3 segundos, se levanta y se acerca a la cámara. Imágenes fotorrealísticas",
    params={
        "model": "wan",
        "duration": 6,
        "aspectRatio": "16:9",
        "audio": True,
        "enhance": True,
        "image": "https://i.ibb.co/CpZvrgGw/doki1.jpg,https://i.ibb.co/gFrRgLQd/doki2.jpg",
    },
)

with open("interpolated.mp4", "wb") as f:
    f.write(video_bytes)
