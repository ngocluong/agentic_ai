import re
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import Literal
from w7_agent.config import settings
from tavily import TavilyClient
llm = ChatGroq(model=settings.llm_model, temperature=settings.temperature, groq_api_key=settings.groq_api_key)

class AgentState(TypedDict):
  topic: str
  search_results: list      # raw Tavily results
  key_points: str           # extracted facts
  gaps: str                 # what's missing after reflection
  sources: list             # URLs cited
  iterations: int           # how many search loops
  report: str               # final markdown report
  review_score: int         # quality score from review_node
  
def receive_topic_node(state: AgentState):
  print('current node: receive_topic_node')
  topic = input('what is your concern topic that want me to write a report for?')
  return { "topic": topic }

def search_web_node(state: AgentState):
  print('current node: search_web_node')
  client = TavilyClient(api_key=settings.tavily_api_key)
  results = client.search(state["topic"], max_results=5)
  return { "search_results": results }
  
def extract_key_points_node(state: AgentState):
  print('current node: extract_key_points_node')
  response = llm.invoke(f"""You are a helpful assistant. Please extract the key facts from search based on topic and search result:
    Topic: {state['topic']},
    Search Result: {state['search_results']}
    In addition here is some gaps that you can based on this if improve the search: {state['gaps']}""")
  return { "key_points": response.content }
  
def reflect_on_gaps_node(state: AgentState):
  print('current node: reflect_on_gaps_node')
  response = llm.invoke(
    f"""You are a helpful assistant. we need to writ a report for the topic: {state['topic']}
    base on key points {state['key_points']}                    
    and report need to be include Overview, Key Finding, Technical Details, Implication and Impact, Conclusion
    Any gaps found based on current key points we have?
    If no gaps found, return only words: [no gaps]""")
  if "[no gaps]" in response.content.lower():
    return { "gaps": '', "iterations": 0 }
  else:
    return { "gaps": response.content, "iterations": state["iterations"] + 1 }

def write_report_node(state: AgentState):
  print('current node: write_report_node')
  response = llm.invoke(
    f"""You are a helpful assistant. Based on those information:
    Topic: {state['topic']},
    Search Result: {state['search_results']}
    Key point: {state['key_points']}
    Please write structured markdown report with 5 sections and based on the following format:
    # {state['topic']} Research Report
    ## 1. Overview
    ## 2. Key Findings  
    ## 3. Technical Details
    ## 4. Implications & Impact
    ## 5. Conclusion
    ## Sources
    - [source 1](url)
    - [source 2](url)
    """)
  return { "report": response.content}

def review_report_node(state: AgentState):
  print('current node: review_report_node')
  response = llm.invoke(f"""
    You are a helpful assistant. Based on those information:
    Topic: {state['topic']},
    Search Result: {state['search_results']}
    Key point: {state['key_points']}
    Please Review this report: {state['report']}. and scores the report quality.
    Return value is only scored from 1 to 10
    """)
  score_match = re.search(r"\d+", response.content)
  review_score = int(score_match.group()) if score_match else 0
  return { "review_score": review_score, "iterations": state["iterations"] + 1 }

def review_gap_edge(state: AgentState) -> Literal["extract_key_points_node", "write_report_node"]:
  print("current edge: gap_check_node")
  
  if(state['gaps'] != '' and state["iterations"] <= 3):
    return('extract_key_points_node')
  else:
    return "write_report_node"
  
  
def review_report_edge(state: AgentState) -> Literal["write_report_node", END]:
  print(f"current edge: review_report_edge with score {state['review_score']}")
  if(state['review_score'] >= 7 or state["iterations"] > 2):
    return END
  else:
    return "write_report_node"
  
agent_builder = StateGraph(AgentState)
agent_builder.add_node(receive_topic_node)
agent_builder.add_node(search_web_node)
agent_builder.add_node(extract_key_points_node)

agent_builder.add_node(reflect_on_gaps_node)
agent_builder.add_node(write_report_node)
agent_builder.add_node(review_report_node)

agent_builder.add_edge(START, "receive_topic_node")
agent_builder.add_edge("receive_topic_node", "search_web_node")
agent_builder.add_edge("search_web_node", "extract_key_points_node")
agent_builder.add_edge("extract_key_points_node", "reflect_on_gaps_node")
agent_builder.add_conditional_edges(
  "reflect_on_gaps_node",
  review_gap_edge,
  ["extract_key_points_node", "write_report_node"]
)
agent_builder.add_edge("write_report_node", "review_report_node")
agent_builder.add_conditional_edges(
  "review_report_node",
  review_report_edge,
  ["write_report_node", END]
)

agent = agent_builder.compile()
if __name__ == "__main__":
  result = agent.invoke({"topic": "", "search_results": [], "key_points": "", "gaps": "", "sources": '', "iterations": 0, "report": '', 'review_score': 0})
  with open(f"{result["topic"].replace(' ', '_')}_report.md", "w") as f:
    f.write(result["report"])