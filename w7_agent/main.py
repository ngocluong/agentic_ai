import sys

def main():
  if len(sys.argv) < 2:
    print("Usage: python -m w7_agent.main <agent>")
    print("Available agents: reflection, parallel, memory, intro")
    return

  agent = sys.argv[1].lower()

  if agent == "reflection":
    from w7_agent.agents.langgraph_reflection import agent as reflection_agent
    topic = input("What topic would you like to discuss? ")
    result = reflection_agent.invoke({"topic": topic, "draft": "", "critique": "", "iteration": 0, "final": "", "quality_score": 0})
    print(result["final"])

  elif agent == "parallel":
    from w7_agent.agents.parallel import agent as parallel_agent
    result = parallel_agent.invoke({})
    print(result["answer"])

  elif agent == "memory":
    from w7_agent.memory.langraph import agent
    # memory agent has its own while loop — just call it directly
    from w7_agent.memory.langraph import run
    run()

  elif agent == "research":
    from w7_agent.agents.research_agent import agent as research_agent
    result = research_agent.invoke({"topic": "", "search_results": [], "key_points": "", "gaps": "", "sources": '', "iterations": 0, "report": '', 'review_score': 0})
    with open(f"{result["topic"].replace(' ', '_')}_report.md", "w") as f:
      f.write(result["report"])
    print(f"Please check result file: {result["topic"].replace(' ', '_')}_report.md")
    
  elif agent == "intro":
    from w7_agent.graph.langraph_intro import agent as intro_agent
    result = intro_agent.invoke({"messages": [], "is_question": False})
    print(result["messages"][-1].content)

  else:
    print(f"Unknown agent: {agent}")
    print("Available: reflection, parallel, memory, intro")

if __name__ == "__main__":
    main()