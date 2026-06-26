import json
import re
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from w5.registry import ToolRegistry
from w5.tools import caculation, format_date, format_text, read_csv

registry = ToolRegistry()
registry.register("calculation", caculation)
registry.register("date", format_date)
registry.register("text", format_text)
registry.register("csv", read_csv)

load_dotenv()    
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def classify_query(query: str) -> str:
    response = llm.invoke(f"""Classify this query into exactly one category.
        Categories: calculation, date, text, csv, general
        Query: {query}
        Return only the category name, nothing else.""")
    return response.content.strip().lower()

def get_input_for_tool(category: str, user_input: str) -> str:
    if category == "calculation":
        response = llm.invoke("get math regestion from"
            f"question: {user_input}"
            "Return only JSON formatted the mathematic notthing else. (e.g { \"expression\": \"1+4*12\" })")
        return response.content.strip().lower()
    elif category == "date":
        response = llm.invoke("get date format from"
            f"question: {user_input}"
            "the valid date format only YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY, day, timestamp"
            "Return only JSON formatted the date format notthing else. (e.g { \"format\": \"YYYY-MM-DD\" })")
        return response.content.strip()
    elif category == "text":
        response = llm.invoke("get test case expected and word to format from"
            f"question: {user_input}"
            "the valid case format only uppercase, lowercase, titlecase, reversed"
            "Return only JSON formatted the word to transfer and case notthing else."
            "(e.g { \"text\": \"text to Upcase\", \"case\": \"uppercase\" })")
        return response.content.strip()
    elif category == "csv":
        response = llm.invoke("get the file path to read from"
            f"question: {user_input}"
            "Return only JSON formatted file path notthing else."
            "(e.g { \"file_path\": \"data/ecommerce.csv\" })")
        return response.content.strip().lower()
    else:
        return ""
    
def routing_agent(user_input: str) -> str:
    category = classify_query(user_input)
    if category == "general":
        return {"success": True, "result": llm.invoke(user_input).content, "error": None}
    
    raw = get_input_for_tool(category, user_input)
    print(f"[debug] raw LLM output: {repr(raw)}")
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return {"success": False, "result": None, "error": f"Could not extract JSON from LLM response: {repr(raw)}"}
    try:
        tool_input = json.loads(match.group())
    except json.JSONDecodeError as e:
        return {"success": False, "result": None, "error": f"JSON parse error: {e} — raw: {repr(match.group())}"}
    return registry.call(category, tool_input)

# user_input = input("May I help you with something?")
# answer = routing_agent(user_input)
# print(answer)


TEST_QUERIES = {
    "What is 25 * 4?": 'calculation',
    "Calculate 100 / 5 + 3": 'calculation',
    "What day is today?": 'date',
    "Give me today's date in YYYY-MM-DD format": 'date',
    "Make 'hello world' uppercase": 'text',
    "Reverse this text: Python is great": 'text',
    "Load my ecommerce.csv file": 'csv',
    "Read the sales data from data.csv": 'csv',
    "What is an AI agent?": "general",
    "Explain LangChain in simple terms": "general"
}

correct = 0
for query in TEST_QUERIES.keys():
    category = classify_query(query)
    print("*"*100)
    print(f"Query: {query[:50]}")
    print(f"Classified as: {category}")
    answer = routing_agent(query)
    print(f"Answer: {answer}")
    if category == TEST_QUERIES[query]:
        correct += 1
    else:
        print(f"expected: {TEST_QUERIES[query]}, got: {category}")

print(f"Score: {correct}/10")

# === REFLECTION ===
# Score: 9/10 — LLM classification is accurate
# What failed: "Explain LangChain" classified as text — LLM saw "LangChain" as something to format
# Fix: add more examples to the classify prompt or add "general" fallback keywords

# LLM classifier vs keyword matcher:
# - LLM: more flexible, handles paraphrasing, costs 1 API call per query
# - Keywords: fast, free, brittle — "What is 25?" wouldn't match "calculate"

# Two-step pipeline (classify → extract args) worked well
# Biggest risk: get_input_for_tool adds latency — 2 LLM calls per query