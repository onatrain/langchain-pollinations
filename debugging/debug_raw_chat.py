import json
import dotenv

from langchain_core.messages import HumanMessage
from langchain_pollinations.chat import ChatPollinations

dotenv.load_dotenv()

model = ChatPollinations(model="openai", temperature=0.2)

res = model.invoke([HumanMessage(content="Responde solo con la palabra: OK")])

print("res.content:", repr(res.content))
print("res.additional_kwargs keys:", sorted(list(res.additional_kwargs.keys())))

raw = res.additional_kwargs.get("raw")
print("raw type:", type(raw))

if raw is not None:
    print(json.dumps(raw, ensure_ascii=False, indent=2))

