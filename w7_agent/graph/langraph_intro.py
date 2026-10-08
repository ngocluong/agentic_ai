from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from w7_agent.config import settings
from typing import Literal

llm = ChatGroq(model=settings.llm_model, temperature=settings.temperature, groq_api_key=settings.groq_api_key)

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]
  is_question: bool
  # add more fields here as the agent grows, e.g.:
  # summary: str

def answer_node(state: AgentState):
  print("current node: answer_node")
  response = llm.invoke(state["messages"])
  return {"messages": [response]}

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
  return {"is_question": "yes" in response.content.strip().lower()}

def should_process(state: AgentState) -> Literal["process_node", END]:
  """Stop the loop if the user wants to exit."""
  print("current Edge: should_process")
  last_message = state["messages"][-1].content.strip().lower()
  if last_message in ("exit", "quit"):
    return END
  return "process_node"

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
  
# Build workflow
agent_builder = StateGraph(AgentState)
# Add nodes
agent_builder.add_node(input_node)
agent_builder.add_node(answer_node)
agent_builder.add_node(process_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "input_node")
agent_builder.add_conditional_edges(
  "input_node",
  should_process,
  ["process_node", END]
)
agent_builder.add_conditional_edges(
  "process_node",
  should_continue,
  ["answer_node", "input_node"]
)
agent_builder.add_edge("answer_node", END)

# Compile the agent
agent = agent_builder.compile()

if __name__ == "__main__":
  result = agent.invoke({"messages": [], "is_question": False})
  print(result["messages"][-1].content)