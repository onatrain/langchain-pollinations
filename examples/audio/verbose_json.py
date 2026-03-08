import dotenv
from langchain_pollinations import STTPollinations

dotenv.load_dotenv()

stt = STTPollinations(response_format="verbose_json")
with open("dialogue.wav", "rb") as fh:
    result = stt.transcribe(fh.read())

print(result.text)
print("Segments:", result.model_extra.get("segments"))
print("Detected language:", result.model_extra.get("language"))