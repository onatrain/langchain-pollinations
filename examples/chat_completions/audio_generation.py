import dotenv, base64

from langchain_core.messages import HumanMessage
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

m = ChatPollinations(
    model="openai-audio",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "mp3"},
)

palabras = "Di en voz alegre: ¡Ya tenemos ChatPollinations! ¡¿Y ahora qué agentes vamos a construir?!"

res = m.invoke([HumanMessage(content=palabras)])

# Texto visible (si existe)
print("text:", res.content)

# Audio (base64) + transcript (cuando el backend lo devuelva así)
audio_obj = res.additional_kwargs.get("audio") or {}
audio_b64 = audio_obj.get("data")
transcript = audio_obj.get("transcript")

print("transcript:", transcript)

if audio_b64:
    with open("simple_audio.mp3", "wb") as f:
        f.write(base64.b64decode(audio_b64))