import dotenv

from langchain_pollinations.image import ImagePollinations

dotenv.load_dotenv()

url1 = "https://i.ibb.co/CpZvrgGw/doki1.jpg"
url2 = "https://i.ibb.co/gFrRgLQd/doki2.jpg"

REFERENCE_IMAGES_URL = f"{url1}|{url2}"

img = ImagePollinations(
    model="seedance",
    image=REFERENCE_IMAGES_URL,
    duration=4,
    aspectRatio="16:9",
    audio=True,
)

data = img.generate("smooth camera movement, cinematic interpolation")

out_file = "mister.mp4"

with open(out_file, "wb") as f:
    f.write(data)

print("OK:", out_file, "bytes:", len(data))
