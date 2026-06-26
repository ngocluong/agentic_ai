from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
import re

load_dotenv()    
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

response = llm.invoke("""If John has 5 pears, 
                      then eats 2, 
                      and buys 5 more, 
                      then gives 3 to his friend, 
                      how many pears does he have?
                      Let's think step by step.
                      answer should be in json format 
                      eg with thinking and result keys""")
response = json.loads(response.content)

print("=====result======")
print(response['result'])
print("=====steps======")
print(response['thinking'])

# === REFLECTION ===
# What did "Let's think step by step" add?
# - Without it: LLM jumps to answer, more likely to make arithmetic errors
# - With it: forces sequential reasoning, each step checks the previous

# When does chain of thought help most?
# - Multi-step math and logic problems
# - Problems where order matters (like this one)
# - Ambiguous questions that need reasoning before answering

# When does it NOT help?
# - Simple factual questions ("What is the capital of France?")
# - Single-step calculations ("What is 2+2?")
# - It adds latency and tokens for no gain on simple tasks

# How does this connect to your agents?
# - Day 13 ReAct loop was chain of thought in action
# - Each "Thought:" step in ReAct is the LLM reasoning before acting
# - Structured JSON output (thinking + result) makes reasoning auditable