from langchain_groq import ChatGroq
from langchain.agents import create_agent
from w5_coffee_agent.agent import tools

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

agent = create_agent(
  model=llm,
  tools=tools,
  system_prompt="""You are a helpful business data analyst. 
  Return your answer in a clear and concise manner, 
  the answer should be in a structured format, valid JSON with keys.
  The response is only in JSON format without wrapping in.""",
)

question = input("what is your question: ")
result = agent.invoke({"messages": [{"role": "user", "content": question}]})
print(result["messages"][-1].content)