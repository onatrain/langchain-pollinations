import dotenv
from langchain_pollinations import STTPollinations

dotenv.load_dotenv()

stt = STTPollinations(response_format="srt")
with open("dialogue.wav", "rb") as fh:
    subtitles = stt.transcribe(fh.read())  # returns str
with open("subtitles.srt", "w") as f:
    f.write(subtitles)