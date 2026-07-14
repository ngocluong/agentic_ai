from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb
import os
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="w6_concepts")
    
def ai_answer(question: str) -> tuple[str, str]:    
    try:
      results = collection.query(query_texts=[question], n_results=5)
    except Exception as e:
      print(f"Error querying ChromaDB: {e}")
      return [], "Error querying ChromaDB. Please check the database connection and query format."

    docs = results["documents"][0] if results["documents"] else []
    
    distances = results["distances"][0] if results["distances"] else [1.0]
    min_distance = min(distances) if distances else 1.0

    low_confidence = min_distance > 1.6
    if not docs or low_confidence:
      facts = "No relevant context found."
    else:
      metadatas = results["metadatas"][0] if results["metadatas"] else []
      facts = "\n\n".join([
        f"[Chunk {i+1}][{os.path.basename(metadatas[i].get('source', 'unknown'))}] {doc}"
        for i, doc in enumerate(docs)
      ])
    
    answer_prompt = PromptTemplate(
      template="""You are a helpful developer assistant.
      Answer using the context below. Cite the chunk number used.
      Use context as primary source. If exact answer isn't there, reason from related concepts but mention you're inferring.
      When listing steps or components, always describe each one fully — not just the name.
      Context: {facts}
      Question: {question}""",
      input_variables=["facts", "question"],
    )
    answer_chain = answer_prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({ "facts": facts, "question": question })
    
    if not answer or len(answer.strip()) < 10:
      answer = answer_chain.invoke({"facts": facts, "question": f"Please provide a complete answer: {question}"})
      if not answer or len(answer.strip()) < 10:
        return facts, "I couldn't generate an answer. Please try rephrasing."
      
    return facts, answer
    
if __name__ == "__main__":
    question = input("what is your question: ")
    facts, answer = ai_answer(question)
    print(f"Context: {facts}")
    print(f"Answer: {answer}")