from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Literal
from uuid import uuid4

load_dotenv()  # Load environment variables from .env file
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]
  is_question: bool
  user_name: str
  current_task: str
  # add more fields here as the agent grows, e.g.:
  # summary: str

def answer_node(state: AgentState):
  print("current node: answer_node")
  response = llm.invoke(state["messages"])
  return {"messages": [response]}

def greet_node (state: AgentState):
  print(f"=== Restored state: {len(state['messages'])} messages in history ===")
  print(f"Hi {state['user_name']}!")
  return {}

def input_node(state: AgentState):
  print("current node: input_node")
  user_input = input("Ask me something: ")
  return {"messages": [{"role": "user", "content": user_input}]}

def process_node(state: AgentState):
  # This node could process the messages, e.g., summarize, analyze, etc.
  # For now, it just returns the state unchanged
  print("current node: process_node")
  last_message = state["messages"][-1].content
  response = llm.invoke(f"""Is the following a valid question? Respond with exactly one word: yes or no.
                    {last_message}""")
  return {
    "is_question": "yes" in response.content.strip().lower(),
    "current_task": last_message
  }

def confirm_node(state: AgentState):
  print("current node: confirm_node")
  last = state["messages"][-1].content
  confirm = input(f"I understood: '{last}'. Is this correct? (yes/no): ")
  return {"is_question": confirm.strip().lower() == "yes"}
  
def should_process(state: AgentState) -> Literal["confirm_node", END]:
  """Stop the loop if the user wants to exit."""
  print("current Edge: should_process")
  last_message = state["messages"][-1].content.strip().lower()
  if last_message in ("exit", "quit"):
    return END
  return "confirm_node"

def should_continue(state: AgentState) -> Literal["answer_node", "input_node"]:
    """Decide if we should continue the loop or turn back to input_node."""

    is_question = state["is_question"]

    # If the LLM makes a tool call, then perform an action
    print(is_question)
    print("current state:", state)
    if is_question:
      return "answer_node"

    # Otherwise, we stop (reply to the user)
    return "input_node"
  
def should_answer(state: AgentState) -> Literal["process_node", "input_node"]:
  """Route after confirmation — yes goes to process, no loops back to input."""
  if state["is_question"]:  # confirm_node sets this to True if user said yes
    return "process_node"
  return "input_node"
  
# Build workflow
agent_builder = StateGraph(AgentState)
# Add nodes
agent_builder.add_node(input_node)
agent_builder.add_node(answer_node)
agent_builder.add_node(process_node)
agent_builder.add_node(greet_node)
agent_builder.add_node(confirm_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "greet_node")
agent_builder.add_edge("greet_node", "input_node")
agent_builder.add_conditional_edges(
  "input_node",
  should_process,
  ["confirm_node", END]
)

agent_builder.add_conditional_edges(
    "confirm_node",
    should_answer,
    ["process_node", "input_node"]
)
agent_builder.add_conditional_edges(
  "process_node",
  should_continue,
  ["answer_node", "input_node"]
)
agent_builder.add_edge("answer_node", END)

checkpointer = InMemorySaver()

# Compile the agent
agent = agent_builder.compile(checkpointer=checkpointer)

# Short-term memory: maps user_name -> thread_id for the lifetime of this
# process only. A user who re-enters the same name in this run reuses their
# existing thread_id (and therefore the checkpointer's saved history for it)
# instead of starting a brand new, disconnected thread.
user_threads: dict[str, str] = {}
user_threads_seen: set[str] = set()  # ← add this

while True:
  name = input("Please enter your name to start the agent (or 'exit' to quit): ").strip()
  if name.lower() in ("exit", "quit"):
    break
  
  thread_id = user_threads.setdefault(name, str(uuid4()))
  is_new_thread = thread_id not in user_threads_seen
  user_threads_seen.add(thread_id)
  initial_state = {"messages": [], "is_question": False, "user_name": name} if is_new_thread else {}
  config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

  result = agent.invoke(initial_state, config=config)
  print(result["messages"][-1].content)
  
  
#   # Compile the agent
# agent = agent_builder.compile()

# result = agent.invoke({"messages": [], "is_question": False})
# print(result["messages"][-1].content)