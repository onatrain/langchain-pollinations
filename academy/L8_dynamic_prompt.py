"""
El agente creado por create_agent() se puede customizar de 3 maneras distintas:
- middleware
- dynamic prompts
- human in the loop
El middleware permite insertar código específico para el agente en puntos clave del loop ReAct:
Request -> before_agent -> before_model -> wrap_model_call -> after_model -> wrap_tool_call -> before_model
                                                                          -> after_agent -> result

before_agent: se usa para setup (files, connections)
before_model: sumarización, guardrails
wrap_model_call: dynamic prompt, model
wrap_tool_call: reintentos, caching
after_model: guardrails
after_agent: teardown

Node hooks: before_agent, before_model, after_model, after_agent
Interceptor hooks: wrap_model_call, wrap_tool_call

Los model call wrappers se pueden usar para seleccionar prompts dinámicamente, y el after_model hook se puede
usar para añadir human in the loop o guardrails en el flujo agéntico.

Selección dinámica de prompts:
A medida que el alcance y duración de las tareas que un agente puede manejar se incrementan, el prompt debe
expandirse para cubrir todas las fases, pasos y contingencias de una tarea. Esto se puede manejar con selección
dinámica de prompts. Los prompts pueden ser seleccionados al vuelo usando el runtime context del estado actual
del agente.
"""

import dotenv
from dataclasses import dataclass

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_pollinations import ChatPollinations
from langgraph.runtime import get_runtime
from langchain.agents.middleware.types import ModelRequest, dynamic_prompt
from langchain.agents import create_agent

dotenv.load_dotenv()


@dataclass
class RuntimeContext:
    is_employee: bool
    db: SQLDatabase


@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query and return results"""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


# Table limits se colocará dependiendo de si se trata de un empleado o no, según el runtime context
system_prompt_template = """Tu eres un cuidadoso analista SQLite.

Reglas:
- Piensa step by step.
- Cuando necesites datos, llama a la herramienta `execute_sql` con UN query SELECT.
- Solo Read-only. Nada de INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limita a 5 filas de salida a menos que el usuario explicitamente solicite otra cosa.
{table_restrictions}
- Si la herramienta retorna 'Error:', revisa el SQL y trata nuevamente.
- Preferentemente usar lista explicita de columnas; evitar SELECT *.
"""


# Utilizar el runtime context y middleware para generar un prompt dinámico
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    if not request.runtime.context.is_employee:
        table_restrictions = "- Limita acceso a estas tablas: Album, Artist, Genre, Playlist, PlaylistTrack, Track."
    else:
        table_restrictions = ""

    return system_prompt_template.format(table_restrictions=table_restrictions)


model = ChatPollinations(
    model="deepseek",
)

# Crear agente pasándole el middleware
agent = create_agent(
    model=model,
    tools=[execute_sql],
    context_schema=RuntimeContext,
    middleware=[dynamic_system_prompt],
)

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

question = "Cuál es la compra más costosa de Frank Harris?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    context=RuntimeContext(is_employee=False, db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

"""
is_employee == True:
La compra más costosa de Frank Harris es la factura con ID **145**, con un total de **$13.86**. Esta es la factura más cara entre todas sus compras.

¿Te gustaría que muestre los detalles específicos de esta factura y los artículos que incluye?

is_employee == False:
La compra más costosa de Frank Harris fue de $13.86 el 23 de septiembre de 2010 (Invoice ID: 145).
"""