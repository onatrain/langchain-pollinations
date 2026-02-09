import dotenv

from langchain_pollinations.image import ImagePollinations

dotenv.load_dotenv()


def ext_from_content_type(ct: str) -> str:
    ct = (ct or "").split(";")[0].strip().lower()
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
    }.get(ct, ".bin")


img = ImagePollinations(model="seedance", duration=6, aspectRatio="16:9", audio=True)

resp = img.generate_response("a taxi drone flying over a city at sunset")
ext = ext_from_content_type(resp.headers.get("content-type", ""))
out_file = f"drone{ext}"

with open(out_file, "wb") as f:
    f.write(resp.content)

print("OK:", out_file, "ctype:", resp.headers.get("content-type"), "bytes:", len(resp.content))
