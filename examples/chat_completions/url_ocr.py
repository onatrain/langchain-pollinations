import dotenv
from langchain_core.messages import HumanMessage
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

m = ChatPollinations(
    model="gemini-fast",  # Solo modelos aceptados en chat completions con función de visión
)

doc_url = "https://i.ibb.co/v48NP6kg/grafica.jpg"

res = m.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "Extrae el texto visible (OCR) y esquematízalo."},
        {"type": "image_url", "image_url": {"url": doc_url}},
    ])
])

print(res.content)