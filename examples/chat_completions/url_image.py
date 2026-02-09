import dotenv

from langchain_core.messages import HumanMessage
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

m = ChatPollinations(
    model="openai",
)

res = m.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "Describe la imagen y enumera 5 objetos visibles."},
        {"type": "image_url", "image_url": {"url": "https://i.ibb.co/JRNjdnPW/groove-Maduro.jpg"}},
    ])
])

print(res.content)