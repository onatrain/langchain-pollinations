import dotenv
from langchain_core.runnables import RunnableLambda
from langchain_pollinations import STTPollinations, ChatPollinations
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

stt = STTPollinations(language="en")
llm = ChatPollinations(model="openai-fast")

pipeline = (
    stt
    | RunnableLambda(lambda r: [HumanMessage(content=f"Summarize: {r.text}")])
    | llm
)

with open("dialogue.wav", "rb") as fh:
    summary = pipeline.invoke(fh.read())
print(summary.content)