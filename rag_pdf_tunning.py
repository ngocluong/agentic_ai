from datetime import datetime
from time import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import markdown
import chromadb
import uuid
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, re
from rag_pdf import ai_answer, invoke_with_retry
import csv
import os

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


def ai_answer_val(question: str, expected: str) -> str:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    facts, answer = ai_answer(question)
    
    eval_prompt = PromptTemplate(
        template="""You are an evaluator scoring a RAG agent's answer.

        Question: {question}
        Expected answer: {expected}
        Actual answer: {answer}

        Score 1-5 based on how well the actual answer matches the expected answer:
        5 = correct and complete
        4 = mostly correct, minor gaps
        3 = partially correct
        2 = mostly wrong
        1 = completely wrong or "I don't know" when the expected answer is known

        Return JSON only:
        {{"score": 5, "reason": "...", "answer_chain": "{answer}"}}
    """, input_variables=["question", "expected", "answer"]
    )
    eval_chain = eval_prompt | llm | StrOutputParser()
    evaluation = invoke_with_retry(eval_chain, {"answer": answer, "facts": facts, "question": question, "expected": expected})
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
    answer, evaluation = ai_answer_val(question, expected)
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
    
filename = "eval_history.csv"
headers = ["Date", "overall_score", "Passed"]

file_exists = os.path.isfile(filename)
with open(filename, mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # If the file did not exist, write the header first
    if not file_exists:
        writer.writerow(headers)
    writer.writerow([datetime.now().strftime("%Y-%m-%d"), f"{sum(scores)/len(scores):.1f}", sum(1 for s in scores if s >= 4)])


# === Day 43 Prompt Tuning Results ===
# Prompt A (baseline):          avg 4.2/5, passed 8/10
# Prompt B+C (rewrite + n=7):   avg 3.9/5, passed 8/10  ← worse
# Prompt D (n=7, no rewrite):   avg 4.2/5, passed 9/10  ← winner

# Winner: Prompt D
# Why: question rewriting lost specific keywords before ChromaDB query
# n_results=7 helped Q8 (ChromaDB question) from 0→4
# Q9 still fails — "formats/output" vocabulary not in README
# Fix for Q9: either update README or rephrase test question

# Key learning: question rewriting can hurt RAG if it removes
# domain-specific terms that match the indexed documents
