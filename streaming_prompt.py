from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
from w5_coffee_agent.agent import tools
load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

def ask_ai(prompt, temperature=0):
  for chunk in llm.stream(prompt):
    if chunk.content:
      yield f"data: {chunk.content}\n\n"
      
def ask_ai_with_tools(prompt, temperature=0):
  agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are a helpful business data analyst. 
    Return your answer in a clear and concise manner, 
    the answer should be in a structured format, valid JSON with keys.
    The response is only in JSON format without wrapping in.""",
  )
  seen_tool_call_ids = set()
  
  for message_chunk, _metadata in agent.stream(
    {"messages": [{"role": "user", "content": prompt}]},
    stream_mode="messages",
  ):
    if message_chunk.content:
      # add a newline before each new tool result chunk
      yield f"data: \n{message_chunk.content}\n\n"
    for tool_call in getattr(message_chunk, "tool_calls", None) or []:
      tool_call_id = tool_call.get("id")
      if tool_call_id and tool_call_id not in seen_tool_call_ids:
        seen_tool_call_ids.add(tool_call_id)
        yield f"tools: {tool_call['name']}\n\n"
