from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import markdown
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb
import uuid
import time
from groq import RateLimitError
from dotenv import load_dotenv

load_dotenv()
NO_ANSWER_PHRASES = ["not in the context", "i don't know", "not mentioned", "no relevant context"]


def invoke_with_retry(runnable, payload, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            return runnable.invoke(payload)
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            wait = float(e.response.headers.get("retry-after", 5))
            print(f"Rate limited, retrying in {wait:.1f}s...")
            time.sleep(wait)

chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="chunk_info")
if collection.count() == 0:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
    )
    with open("week_3_README.md", "r") as f:
        html = markdown.markdown(f.read())

    docs = text_splitter.create_documents([html])
    collection.add(
        documents=[doc.page_content for doc in docs],
        metadatas=[{"source": "readme.md"} for _ in docs],
        ids=[str(uuid.uuid4()) for _ in docs],
    )
    print(f"Indexed {len(docs)} chunks.")


def ai_answer(question: str) -> str:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    try:
        results = collection.query(query_texts=[question], n_results=7)
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return [], "Error querying ChromaDB. Please check the database connection and query format."
    
    docs = results["documents"][0] if results["documents"] else []

    distances = results["distances"][0] if results["distances"] else [1.0]
    min_distance = min(distances) if distances else 1.0
    low_confidence = min_distance > 1.2

    if not docs or low_confidence:
        facts = "No relevant context found."
    else:
        facts = "\n\n".join([f"[Chunk {i+1}] {doc}" for i, doc in enumerate(docs)]) 
    
    def ask(q: str) -> str:
        answer_prompt = PromptTemplate(
            template="""You are a helpful developer assistant.
            Answer using the context below. Cite the chunk number used.
            Use context as primary source. If exact answer isn't there, reason from related concepts but mention you're inferring.
            Context: {facts}
            Question: {question}""",
            input_variables=["facts", "question"],
        )

        answer_chain = answer_prompt | llm | StrOutputParser()
        answer = invoke_with_retry(answer_chain, {"facts": facts, "question": q})

        return answer

    # Fallback 4 — ask LLM
    answer = ask(question)

    # Fallback 3 — empty response → retry once
    if not answer or len(answer.strip()) < 10:
        answer = ask(f"Please provide a complete answer: {question}")
        if not answer or len(answer.strip()) < 10:
            return facts, "I couldn't generate an answer. Please try rephrasing."

    # Fallback 4 — "I don't know" is valid, return as-is
    if any(p in answer.lower() for p in NO_ANSWER_PHRASES):
        return facts, f"I'm not sure — {answer}"

    return facts, answer

if __name__ == "__main__":
    question = input("what is your question: ")
    facts, answer = ai_answer(question)
    print(f"Context: {facts}")
    print(f"Answer: {answer}")

# === REFLECTION ===
# Did citation work? Yes — LLM referenced "Chunk 2" 
# But the chunk boundary cut the answer awkwardly between two chunks
# The model self-corrected by saying "described in the section before it"

# Why does this happen?
# - chunk_size=500 with chunk_overlap=50 isn't always enough to avoid 
#   splitting a single concept across 2 chunks
# - The langchain_two_agent description in the README is a multi-sentence 
#   block that straddled the 500-char boundary

# How to fix in production:
# - Increase chunk_overlap to 100-150 for documents with dense explanations
# - Or chunk by markdown headers (split on ## instead of raw character count)
#   so each chunk = one complete section

# Key learning:
# Citation accuracy depends on chunk quality, not just retrieval quality
# Even with correct retrieval, bad chunk boundaries create confusing citations