from mini_project_agent import chroma_search, generate_report, llm_answer


def is_research_request(question):
  research_keywords = ["research", "tell me about", "what is", "explain", "overview of"]
  return any(keyword in question.lower() for keyword in research_keywords)

while True:
    question = input("What Can I help: ")
    if question.lower() in ["exit", "quit"]:
        break
    facts = chroma_search(question)
    if is_research_request(question):
        print(generate_report(question, facts))
    else:
        print(llm_answer(question, facts).content)
