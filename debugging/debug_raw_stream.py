import dotenv
from langchain_core.messages import HumanMessage
from langchain_pollinations.chat import ChatPollinations

dotenv.load_dotenv()

model = ChatPollinations(model="openai", temperature=0.2)

for i, chunk in enumerate(model.stream([HumanMessage(content="Di: hola")])):
    print("i=", i, "type=", type(chunk))
    print("repr=", repr(chunk))

    content = getattr(chunk, "content", None)
    if isinstance(content, str) and content:
        print("chunk.content:", repr(content))

    msg = getattr(chunk, "message", None)
    if msg is not None:
        msg_content = getattr(msg, "content", None)
        if isinstance(msg_content, str) and msg_content:
            print("chunk.message.content:", repr(msg_content))

    if i >= 30:
        break

