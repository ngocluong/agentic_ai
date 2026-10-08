from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import chromadb
import uuid

load_dotenv()

client = TavilyClient(os.getenv("TAVILY_API_KEY"))
# runs locally, saves to disk in ./chroma folder
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="research_collection")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def chroma_search(question):
  chromaresults = collection.query(
    query_texts=[question],
    n_results=3
  )
  if chromaresults['documents'] == [[]]:
    facts = tavily_search(question)
    for result in facts:
      summarized = summary(result['content'])
      collection.add(
        documents=[question + ". Answer:" + summarized],
        metadatas=[{"question": question, "answer": summarized}],
        ids=str(uuid.uuid4())
      )
  else:
    facts = chromaresults['documents']  
    
  return facts
  
def tavily_search(prompt):
  response = client.search(
      query=prompt,
      search_depth="advanced",
      max_results=3
  )
  return response['results']

def llm_answer(question, facts):
  prompt = PromptTemplate(
      template="""You are a helpful agent that answers the question based on the provided facts. 
      Answer the user question: {question}
      Based on the following facts: {facts}""",
      input_variables=["question", "facts"]
  )
  chain = prompt | llm 
  return chain.invoke({"question": question, "facts": facts})

def summary(facts):
  prompt = PromptTemplate(
      template="""You are a helpful agent that summarizes the provided facts. 
      Summarize the following facts: {facts}""",
      input_variables=["facts"]
  )
  chain = prompt | llm
  return chain.invoke({"facts": facts}).content

def generate_report(question, facts):
    prompt = PromptTemplate(
      template=
        """Based on these facts: {facts}
        Generate a structured report about: {question}

        Format exactly as:
        ## Summary
        (2 sentences)

        ## Key Features  
        (bullet points)

        ## Use Cases
        (bullet points)

        ## Verdict
        (1 sentence — is it worth learning?)""",
      input_variables=["question", "facts"]
    )
    chain = prompt | llm
    return chain.invoke({"question": question, "facts": facts}).content