import dotenv
from langchain_core.messages import HumanMessage
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

m = ChatPollinations(
    model="gemini-fast",
)

url = "http://www.dropbox.com/scl/fi/vgtfqyk2ojnbhnhstyrrp/dron.mp4?dl=1"

prompt = HumanMessage(content=[
    {"type": "text", "text": "Mira este video y dime qué ocurre en la escena principal."},
    {"type": "video_url", "video_url": {"url": url, "mime_type": "video/mp4"}},
])

res = m.invoke([prompt])

print(res.content)