import dotenv
from langchain_pollinations.image import ImagePollinations

dotenv.load_dotenv()

prompt = "combine the images creating a cool composition in anime style"

url1 = "https://i.ibb.co/vvrkV5Xf/tigre.jpg"
url2 = "https://i.ibb.co/qMN4Lhh8/casita.jpg"

REFERENCE_IMAGES_URL = f"{url1}|{url2}"

params = {
    "model": "flux",
    "image": REFERENCE_IMAGES_URL,
    "enhance": True,
}

img = ImagePollinations()
data = img.generate(prompt, params=params)

with open("img2img.jpg", "wb") as f:
    f.write(data)
