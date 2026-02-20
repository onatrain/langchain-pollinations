from langchain_pollinations import ChatPollinations, PollinationsAPIError
from langchain_core.messages import HumanMessage

try:
    llm = ChatPollinations(model="gemini", api_key="anyway")
    res = llm.invoke([HumanMessage(content="Hello")])
    print(res.content)
except PollinationsAPIError as e:
    if e.is_auth_error:
        print("Check your POLLINATIONS_API_KEY.")
    elif e.is_validation_error:
        print(f"Bad request: {e.details}")
    elif e.is_server_error:
        print(f"Server error {e.status_code} – consider retrying.")
    else:
        print(e.to_dict())
