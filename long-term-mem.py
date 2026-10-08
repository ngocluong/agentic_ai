import uuid

from dotenv import load_dotenv
from openai import OpenAI
import os

import chromadb

load_dotenv()  # Load environment variables from .env file

# runs locally, saves to disk in ./chroma folder
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="history_collection")

client = OpenAI(
  api_key=os.getenv("GROQ_API_KEY"),
  base_url="https://api.groq.com/openai/v1",
)

messages = [
  {
    "role": "system",
    "content": "You are a helpful assistant."
  }
]
user_name="Default User"

def ai_answer(question):
  chromaresults = collection.query(
    query_texts=[question],
    n_results=2,
    where={"user": user_name} 
  )
  
  facts = chromaresults['documents'][0] if chromaresults['documents'] else []
  messages[0]["content"] = f"""You are a helpful assistant.
  Relevant facts from past conversations: {facts}"""
  
  messages.append({"role": "user", "content": question})

  response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,  # send full history every time
  )
  
  answer = response.choices[0].message.content
  messages.append({"role": "assistant", "content": answer})  # add this line
  
  collection.add(
    documents=[question + " " + answer],  # could also store question and answer separately
    metadatas=[{"question": question, "answer": answer, "user": user_name}],
    ids=f'{user_name}_{uuid.uuid4()}'
  )
  
  return answer

user_name = input("Before we start, what is your name? ")

while True:
  question = input("What Can I help: ")
  if question.lower() in ["exit", "quit"]:
    print("Exiting...")
    break
  result = ai_answer(question)
  print(result)
  
