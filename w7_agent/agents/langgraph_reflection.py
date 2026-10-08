from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import Literal
from w7_agent.config import settings

llm = ChatGroq(model=settings.llm_model, temperature=settings.temperature, groq_api_key=settings.groq_api_key)

class AgentState(TypedDict):
  draft: str
  critique: str
  iteration: int
  final: str       
  topic: str
  quality_score: int
  
def generate_node(state: AgentState):
  response = llm.invoke(
    f"""You are a helpful assistant. Please generate a draft based on the following topic:
    Topic: {state['topic']}
    """)
  print(f"generate_node: Draft ready ({len(response.content)} chars)")
  return { "draft": response.content, "iteration": 0 }

def critique_node(state: AgentState):
  response = llm.invoke(
    f"""You are a helpful assistant. Please critique the following draft and topic then provide suggestions for improvement:
    {state['draft']}
    Topic: {state['topic']}
    Please provide your critique in a single paragraph with specific, actionable feedback.""")
  print(f"critique_node: {response.content}")
  return { "critique": response.content }

def revise_node(state: AgentState):
  print("current node: revise_node")
  response = llm.invoke(
    f"""You are a helpful assistant. Please revise the following draft based on the critique provided:
    Draft: {state['draft']}
    Critique: {state['critique']}
    Please provide your revised draft""")
  print(f"revise_node: ({len(response.content)} chars)")
  return { "draft": response.content, "iteration": state["iteration"] + 1 }

def quality_check_node(state: AgentState):
  print("current node: quality_check_node")
  response = llm.invoke(
    f"""You are a helpful assistant. Please evaluate the quality of the following draft and score from 1 to 10, where 10 is the highest quality:
    Draft: {state['draft']}. return only the score as a single number, no other text.
    """)
  print(f"quality_check_node: {response.content}")
  if int(response.content.strip()) >= 7:
    return { "final": state['draft'], "quality_score": int(response.content.strip()) }
  else:
    return { "quality_score": int(response.content.strip()) }

def should_process(state: AgentState) -> Literal["revise_node", END]:
  print("current Edge: should_process")
  if state["quality_score"] >= 7 or state["iteration"] >= 3:
    print("*" * 40)
    print(f"Final draft accepted with score {state['quality_score']}.")
    print(f"Final draft: {state['draft']}")
    return END
  else:
    return "revise_node"

agent_builder = StateGraph(AgentState)
agent_builder.add_node(generate_node)
agent_builder.add_node(critique_node)
agent_builder.add_node(revise_node)
agent_builder.add_node(quality_check_node)

agent_builder.add_edge(START, "generate_node")
agent_builder.add_edge("generate_node", "critique_node")
agent_builder.add_edge("critique_node", "revise_node")
agent_builder.add_edge("revise_node", "quality_check_node")
agent_builder.add_conditional_edges(
  "quality_check_node",
  should_process,
  ["revise_node", END]
)
agent = agent_builder.compile()
if __name__ == "__main__":
  topic = input("What topic would you like to ask? ")
  agent.invoke({"topic": topic, "draft": "", "critique": "", "iteration": 0, "final": "", "quality_score": 0})
