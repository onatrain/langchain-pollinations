import dotenv
from langchain_core.messages import HumanMessage
from langchain_pollinations import ChatPollinations
from utils import convert_to_json

dotenv.load_dotenv()

m = ChatPollinations(
    model="gemini-fast",
    tools=[{"type": "url_context"}],
)

url = "https://www.dropbox.com/scl/fi/0yhqcu85x7mf5xkah7czk/mary.mp3?dl=1"

prompt = HumanMessage(content=f"Utiliza la herramienta url_context para acceder y transcribir el audio almacenado en: {url}")

res = m.invoke([prompt])

response = convert_to_json(res.model_dump())
print(response)