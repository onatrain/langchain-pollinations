import dotenv
from langchain_core.messages import HumanMessage
from langchain_pollinations import ChatPollinations
from utils import encode_to_base64

dotenv.load_dotenv()

audio_b64 = encode_to_base64("audio.mp3")

m = ChatPollinations(
    model="openai-audio",
    modalities=['text'],
)

msg = HumanMessage(
    content=[
        {"type": "text", "text": "Transcribe este audio. Devuelve solo el texto."},
        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
    ]
)

res = m.invoke([msg])
print(res.content)