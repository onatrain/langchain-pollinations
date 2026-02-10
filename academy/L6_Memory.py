"""
Para hacer que el agente recuerde la conversación, hay que agregarle memoria via LangGraph.
La memoria persistirá los mensajes y el estado entre llamadas al agente.
El create_agent() de LangChain corre bajo LangGraph. LangGraph expone un objeto Runtime con lo siguiente:
1- Contexto: información estática como user id, db connections, u otras dependencias para una invocación de agente.
2- Store: una instancia BaseStore usada para memoria a largo plazo.
3- Stream writer: un objeto usado para streamear información via el modo stream "custom".

Se puede acceder a la información runtime en las tools, así como también via custom agent middleware.
La memoria a corto plazo se logra mediante checkpointing.

La memoria es muy importante para los agentes, especialmente para los que se envuelven en largas conversaciones que
podrían ser interrumpidas o pausadas. Para tener interacciones productivas, el agente debe recordar cosas previas
sobre el estado de la conversación. La memoria de corto plazo toma forma en mensajes persistentes o estado del agente
entre invocaciones del agente.

"""
import dotenv

from dataclasses import dataclass

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langchain.agents import create_agent
from langchain_pollinations import ChatPollinations
from langgraph.checkpoint.memory import InMemorySaver

dotenv.load_dotenv()

db = SQLDatabase.from_uri("sqlite:///Chinook.db")


@dataclass
class RuntimeContext:
    db: SQLDatabase


@tool
def execute_sql(query: str) -> str:
    """Ejecuta un SQLite query y devuelve el resultado"""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


system_prompt = """You are a carefull SQLite analyst.

Rules:
- Think step-by-step
- When you need data, call the tool `execute_sql` with ONE SELECT query
- Read-only only. No INSERT/UPDATE/DELETE/DROP/TRUNCATE/ALTER/CREATE/REPLACE.
- Limit to 5 rows of output unless the user explicitly asks otherwise.
- If the tool returns `Error:`, revise the Sql and try again.
- Prefer explicit column lists. Avoid SELECT *.
"""

model = ChatPollinations(
    model="deepseek",
)

agent = create_agent(
    model=model,
    tools=[execute_sql],
    system_prompt=system_prompt,
    context_schema=RuntimeContext,
    checkpointer=InMemorySaver(),
)

question = "Habla Frank Harris. Cuál fue el total de mi última factura?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    {"configurable": {"thread_id": "1"}},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

print("=" * 100)

question = "Cuáles fueron todos los títulos?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    {"configurable": {"thread_id": "1"}},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
