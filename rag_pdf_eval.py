from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import markdown
import chromadb
import uuid
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, re

load_dotenv()
TEST_CASES = [
    {
        "question": "What does short-term-mem.py do?",
        "expected": "A conversational assistant that remembers the current session by storing conversation history in a list and passing it back to the LLM on each call."
    },
    {
        "question": "Where does long-term-mem.py store conversation history?",
        "expected": "ChromaDB, a local vector store, with the username as a metadata key."
    },
    {
        "question": "What happens to short-term memory when the script exits?",
        "expected": "Memory disappears — it is not persistent across sessions."
    },
    {
        "question": "What is the purpose of langchain_translation.py?",
        "expected": "A two-stage LangChain pipeline that first summarises an input sentence, then translates the summary into French."
    },
    {
        "question": "What does the pipe operator do in langchain_translation.py?",
        "expected": "Connects the summary chain and translation chain so the output of one becomes the input of the next."
    },
    {
        "question": "How many agents does langchain_two_agent have and what do they do?",
        "expected": "Two agents — one plans by breaking a goal into steps, the other executes each step using tools."
    },
    {
        "question": "What search tool does mini_project_w3.py use?",
        "expected": "Tavily for real-time web search."
    },
    {
        "question": "What does mini_project_w3.py use ChromaDB for?",
        "expected": "As a semantic cache to skip repeat searches — if relevant facts already exist, they are used directly without searching again."
    },
    {
        "question": "What formats does mini_project_w3.py output?",
        "expected": "Either answers a question in free-form Q&A or formats a structured report depending on what the user asked."
    },
    {
        "question": "What is the key takeaway from long-term-mem.py?",
        "expected": "ChromaDB as a local vector store enables LLM memory that persists beyond a single process."
    },
]
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="chunk_info")

# Only index if the collection is empty to avoid duplicates on re-runs
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
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    question = llm.invoke(f"Rephase the question in a concise way: {question}").content
    results = collection.query(query_texts=[question], n_results=5)
    print("=====DEBUG LOG=====")
    print(f"results: {results}")
    docs = results["documents"][0] if results["documents"] else []

    if not docs:
      facts = "No relevant context found."
    else:
      facts = "\n\n".join([f"[Chunk {i+1}] {doc}" for i, doc in enumerate(docs)])
    
    answer_prompt = PromptTemplate(
        template="""You are a helpful developer assistant.
          Answer this question using the context below. Always cite which chunk number you used.
          If the answer isn't in the context, say so.
          Context: {facts}
          Question: {question}""",
        input_variables=["facts", "question"]
    )
    answer_chain = answer_prompt | llm | StrOutputParser()
    
    eval_prompt = PromptTemplate(
        template="""You are a helpful developer assistant.
        Is "{answer}" factually correct per {facts} with question {question}? Score 1-5 + reason and print again the answer.
        return as json example:
        {{"answer_chain": answer from answer_chain, "score": 5, "reason": "The answer is factually correct."}}
        """,
        input_variables=["answer", "facts", "question"]
    )
    eval_chain = eval_prompt | llm | StrOutputParser()

    answer = answer_chain.invoke({"facts": facts, "question": question})
    evaluation = eval_chain.invoke({"answer": answer, "facts": facts, "question": question})
    return answer, evaluation

def parse_eval(text: str) -> dict:
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {"score": 0, "reason": "parse error", "answer_chain": ""}

scores = []
failures = []
for test_case in TEST_CASES:
    question = test_case["question"]
    expected = test_case["expected"]
    answer, evaluation = ai_answer(question)
    eval_data = parse_eval(evaluation)
    score = eval_data.get("score", 0)
    scores.append(score)
    if score < 4:
        failures.append({"question": question, "score": score})
    print(f"Q: {question}")
    print(f"Score: {score}/5 — {eval_data.get('reason', '')[:80]}")
    print("---")

print("\n=== EVALUATION REPORT ===")
print(f"Avg score:     {sum(scores)/len(scores):.1f}/5")
print(f"Passed (≥4):   {sum(1 for s in scores if s >= 4)}/{len(TEST_CASES)}")
print(f"Failed (<4):   {len(failures)}/{len(TEST_CASES)}")
for f in failures:
    print(f"  ❌ {f['question'][:55]} → {f['score']}/5")

# === EVALUATION REPORT ===
# Score: 8/10 passed (≥4/5), avg 4.0/5

# ✅ What worked well (8 questions):
# - short-term-mem, langchain_translation, langchain_two_agent, mini_project search tool
#   all retrieved correct chunks and answered accurately

# ❌ Failure 1: "What does mini_project_w3.py use ChromaDB for?" → 1/5
# Root cause: chunk boundary split the mini_project section across chunks
# The ChromaDB caching description ended up in a chunk that retrieval missed
# Fix: increase chunk_overlap from 50 → 150, or split by ## markdown headers

# ❌ Failure 2: "What is the key takeaway from long-term-mem.py?" → 1/5  
# Root cause: "key takeaway" is a section heading concept — the embedding
# similarity between the question and the actual "Key takeaway" bullet
# was weak because the phrasing doesn't match the README section content
# Fix: add the section heading text into each chunk so retrieval finds it

# Key learning:
# LLM-as-judge is reliable when the answer is clearly in the retrieved chunks
# It breaks down when retrieval fails — garbage in, garbage out
# Evaluation score reflects BOTH retrieval quality AND answer quality
# You can't improve the score without first fixing the chunking strategy

# Next step: fix the 2 failures by tuning chunk_overlap and re-running

# === DEBUG LOG - fixes ===
# Before: 8/10 passed, avg 4.0/5
# After:  10/10 passed (on 2 failing cases), avg 4.5/5

# Fix 1: chunk_overlap 50 → 150
#   Result: ChromaDB description stayed in same chunk as mini_project heading
#   Q8 improved from 1/5 → 4/5

# Fix 2: n_results 3 → 5
#   Result: more context gave LLM enough to find key takeaway
#   Q10 improved from 1/5 → 5/5

# Fix 3: LLM question rewriting
#   Result: more precise embedding query, better chunk matching

# Key lesson: small parameter changes (overlap, n_results) have large impact
# Always re-run full eval after any change to measure actual improvement