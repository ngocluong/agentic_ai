from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from ai_businness_agent import revenue_by_country, revenue_by_country_name, revenue_by_segment

# 1. Wrap imported functions as tools
revenue_by_country = tool(revenue_by_country)
revenue_by_country_name = tool(revenue_by_country_name)
revenue_by_segment = tool(revenue_by_segment)

tools = [revenue_by_country, revenue_by_country_name, revenue_by_segment]

# 2. Initialize the model
llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

# 3. Create the ReAct agent
agent = create_agent(
  model=llm,
  tools=tools,
  system_prompt="""You are a helpful business data analyst. 
  Return your answer in a clear and concise manner, 
  the answer should be in a structured format, valid JSON with keys.
  The response is only in JSON format without wrapping in.""",
)

sentence = input("what is your question? ")
# 4. Invoke the agent
result = agent.invoke({"messages": [("human", sentence)]})
print(result["messages"][-1].content)
