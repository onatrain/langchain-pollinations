import dotenv

from langchain_pollinations.image import ImagePollinations

dotenv.load_dotenv()

prompt = "a cute black and brown German Shepherd dog sleeping on a pillow. Vivid colors image, photorealistic."
out_file = "cute_dog.jpg"

img = ImagePollinations(model="klein")
resp = img.generate_response(prompt)
data = resp.content

with open(out_file, "wb") as f:
    f.write(data)

print("OK:", out_file, "bytes:", len(data))

print(resp.headers)