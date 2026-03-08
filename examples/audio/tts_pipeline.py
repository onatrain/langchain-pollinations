import dotenv
from langchain_core.runnables import RunnableLambda
from langchain_pollinations import ChatPollinations, TTSPollinations

dotenv.load_dotenv()

llm = ChatPollinations(model="openai-fast")
tts = TTSPollinations(voice="shimmer")

pipeline = llm | RunnableLambda(lambda msg: msg.content) | tts
audio = pipeline.invoke("Resume el ciclo del agua en tres oraciones.")
with open("water_cycle.mp3", "wb") as f:
    f.write(audio)
