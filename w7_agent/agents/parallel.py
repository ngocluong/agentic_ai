from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
import time
from w7_agent.config import settings

llm = ChatGroq(model=settings.llm_model, temperature=settings.temperature, groq_api_key=settings.groq_api_key)

class AgentState(TypedDict):
  topic: str
  academic_topics: str
  newest_topics: str
  answer: str
  
def topic_node(state: AgentState):
  topic = input("what you want to discuss? ")
  if topic.lower() in ("exit", "quit"):
      return {"topic": ""}
  return {"topic": topic}
  
def newest_topic_searcher_node(state: AgentState):
  print("current node: newest_topic_searcher_node")
  response = llm.invoke(
    f"""You are a helpful assistant. Please provide a list of the most recent and relevant news in the field of {state['topic']}. 
    Please return the research as a numbered list, with research on a new line.
    """)
  print(f"newest_topic_searcher_node: ({len(response.content)} chars)")
  return { "newest_topics": response.content }

def academic_topic_searcher_node(state: AgentState):
  print("current node: academic_topic_searcher_node")
  response = llm.invoke(
    f"""You are a helpful assistant. Please provide a list of the technical and academic in the field of {state['topic']}. 
    Please return the research as a numbered list, with research on a new line.
    """)
  print(f"academic_topic_searcher_node: ({len(response.content)} chars)")
  return { "academic_topics": response.content }

def join_node(state: AgentState):
  print("current node: join_node")
  response = llm.invoke(
    f"""You are a helpful assistant. From recent news and academic research, provide summary for topic {state['topic']}. 
    recent news: {state['newest_topics']}, academic research: {state['academic_topics']}
    """)
  print(f"join_node: ({len(response.content)} chars)")
  return { "answer": response.content }

def route_after_topic(state: AgentState) -> list:
  if not state["topic"]:
    return [END]
  return ["newest_topic_searcher_node", "academic_topic_searcher_node"]
  
agent_builder = StateGraph(AgentState)
agent_builder.add_node(topic_node)
agent_builder.add_node(newest_topic_searcher_node)
agent_builder.add_node(academic_topic_searcher_node)
agent_builder.add_node(join_node)

agent_builder.add_edge(START, 'topic_node')
agent_builder.add_conditional_edges(
    'topic_node',
    route_after_topic,
)
# ← no extra edge here
agent_builder.add_edge('newest_topic_searcher_node', 'join_node')
agent_builder.add_edge('academic_topic_searcher_node', 'join_node')
agent_builder.add_edge('join_node', END)

agent = agent_builder.compile()
if __name__ == "__main__":
  start = time.time()
  result = agent.invoke({})
  duration = time.time() - start
  print(result["answer"])
  print(f"total exec time: {round(duration, 4)}")

