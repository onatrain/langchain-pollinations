import dotenv
from langchain_core.messages import HumanMessage
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

frame_urls = [
    "https://i.ibb.co/CpZvrgGw/doki1.jpg",
    "https://i.ibb.co/gFrRgLQd/doki2.jpg",
]

content = [{"type": "text", "text": "Estas imágenes son frames de un video. Inventa una historia que describa los cambios entre frames."}]
content += [{"type": "image_url", "image_url": {"url": u}} for u in frame_urls]

m = ChatPollinations(
    model="openai",
)

res = m.invoke([HumanMessage(content=content)])
print(res.content)