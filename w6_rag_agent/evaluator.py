from test_case import TEST_CASES
from agent import ai_answer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import csv, os
from datetime import date

HISTORY_FILE = "w6_rag_agent/eval_history_w6.csv"
headers = ["date","overall_score","passed","q1","q2","q3","q4","q5","q6","q7","q8","q9","q10"]

llm = ChatGroq(
  model="openai/gpt-oss-120b",
  temperature=0,
  model_kwargs={"response_format": {"type": "json_object"}}
)

def eval_ai_answer(question: str, expected: str) -> tuple[str, dict]:
  facts, answer = ai_answer(question)
  eval_prompt = PromptTemplate(
    template="""You are an evaluator scoring a AI agent's answer.
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
    {{"score": 5, "reason": "..."}}
    """,
    input_variables=["question", "expected", "answer"],
  )
  eval_chain = eval_prompt | llm | JsonOutputParser()
  evaluation = eval_chain.invoke({"answer": answer, "question": question, "expected": expected})
  return answer, evaluation

# ── Run evaluation ─────────────────────────────────────────────────────────
scores   = []
failures = []

for i, test_case in enumerate(TEST_CASES, 1):
  question = test_case["question"]
  expected = test_case["expected"]
  answer, evaluation = eval_ai_answer(question, expected)
  score = evaluation.get("score", 0)
  scores.append(score)
  if score < 4:
      failures.append({"question": question, "score": score})
  print(f"Q{i}: {question[:60]}")
  
  
  print(f"  Score: {score}/5 — {evaluation.get('reason', '')[:75]}")

row = [date.today().isoformat(), round(sum(scores)/len(scores), 2),
  sum(1 for s in scores if s >= 4)] + scores

file_exists = os.path.isfile(HISTORY_FILE)
with open(HISTORY_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(headers)
    writer.writerow(row)

print("\n=== EVALUATION REPORT ===")
print(f"Avg score:   {sum(scores)/len(scores):.1f}/5")
print(f"Passed (≥4): {sum(1 for s in scores if s >= 4)}/{len(scores)}")
for f in failures:
    print(f"  ❌ {f['question'][:55]} → {f['score']}/5")
print(f"\nSaved to {HISTORY_FILE}")