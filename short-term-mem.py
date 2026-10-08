from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()  # Load environment variables from .env file

client = OpenAI(
  api_key=os.getenv("GROQ_API_KEY"),
  base_url="https://api.groq.com/openai/v1",
)

messages = [
  {
    "role": "system",
    "content": "You are a helpful assistant. You remember everything said in this conversation."
  }
]

def ai_answer(question):
  # Add user message to history
  messages.append({"role": "user", "content": question})
  
  response = client.chat.completions.create(
      model="openai/gpt-oss-120b",
      messages=messages,  # send full history every time
  )
  
  answer = response.choices[0].message.content
  
  # Add assistant reply to history so next turn remembers it
  messages.append({"role": "assistant", "content": answer})
  
  return answer    

while True:
  question = input("What Can I help: ")
  if question.lower() in ["exit", "quit"]:
    print("Exiting...")
    break
  result = ai_answer(question)
  print(result)
  
  
  # === REFLECTION ===
# How does short-term memory work here?
# - messages list grows with every turn
# - full history sent to LLM on every API call
# - LLM natively understands the role/content format

# What is the limitation of this approach?
# - memory disappears when the script exits (not persistent)
# - long conversations = more tokens = higher cost + slower
# - no way to recall facts from a previous session

# What's next (Day 18)?
# - store facts in a vector database (ChromaDB)
# - retrieve relevant facts on each new question
# - memory survives across separate script runs