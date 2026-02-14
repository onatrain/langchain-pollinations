import dotenv
from langchain_core.messages import HumanMessage
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

frame_urls = [
    "https://i.ibb.co/CpZvrgGw/doki1.jpg",
    "https://i.ibb.co/gFrRgLQd/doki2.jpg",
]

content = [{"type": "text", "text": "Inventa una historia con el personaje de las fotos y sus acciones."}]
content += [{"type": "image_url", "image_url": {"url": u}} for u in frame_urls]

m = ChatPollinations(
    model="openai-large",
)

res = m.invoke([HumanMessage(content=content)])
print(res.content)