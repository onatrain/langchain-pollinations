import dotenv

from langchain_pollinations import ImagePollinations

dotenv.load_dotenv()

gen = ImagePollinations()
video_bytes = gen.generate(
    "Vocalista en concierto cantando y ejecutando groove metal estilo Sepultura, cámara estable",
    params={
        "model": "wan",
        "duration": 4,
        "aspectRatio": "16:9",
        "audio": True,
        "image": "https://i.ibb.co/JRNjdnPW/groove-Maduro.jpg",  # Debe ser un url
    },
)

with open("framed.mp4", "wb") as f:
    f.write(video_bytes)