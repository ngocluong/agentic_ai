from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()  # Load environment variables from .env file

client = OpenAI()
sales_data = {"US_revenue": 12000, "CA_revenue": 8000, "top_category": "Electronics"}


def build_prompt(question, options=None):
    options = options or {}
    role = options.get("role", "data analyst")
    data = json.dumps(sales_data)
    return f"""
    "Using Data: " + {data}
    You are a {role}. 
    {question}
    Return your answer in a clear and concise manner.
    the answer should be in a structured format, valid JSON with keys.
    """

def ask_ai(prompt, temperature=0):
    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

def parse_json_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw output:")
        print(response)
        return None
    
question = input("Ask: ")
prompt = build_prompt(question, options={"role": "senior data analyst"})
answer = ask_ai(prompt, temperature=0)
print(parse_json_response(answer))


# assignent: build ask_ai and build_prompt functions,
# then test with 3 different prompts 
# and 2 runs each to see reliability and consistency of outputs

# answer = ask_ai("Summarize sales performance")
# print("=== Prompt A ===")
# print(answer)

# print("\n=== Prompt B ===")
# answer = ask_ai("Summarize in 3 sentences highlighting top trend", options={"role": "senior data analyst"})
# print(answer)

# print("\n=== Prompt C ===")
# answer = ask_ai("Analyze sales. Return: 1) Top metric 2) Biggest risk 3) Action. Bullets.")
# print(answer)

# print("\n=== Prompt C ===")
# answer = ask_ai("Analyze sales. Return: 1) Top metric 2) Biggest risk 3) Action. Bullets.")
# print(answer)

# print("\n=== Prompt C ===")
# answer = ask_ai("Analyze sales. Return: 1) Top metric 2) Biggest risk 3) Action. Bullets.", options={"temperature": 1})
# print(answer)

# print("\n=== Prompt C ===")
# answer = ask_ai("Analyze sales. Return: 1) Top metric 2) Biggest risk 3) Action. Bullets.", options={"temperature": 1})
# print(answer)

# === REFLECTION ===
# Which prompt was best? Prompt C — structured format forced a useful output even with no real data
# Which was most reliable? Prompt C at temperature=0 — nearly identical both runs
# Which was least reliable? temperature=1 — different wording/order each run
# Key learning: vague prompts (A) give vague answers; structure beats role for consistency

# answer = ask_ai("""
# Analyze this sales data.

# Return ONLY valid JSON.

# Format:
# {
#   "top_metric": "...",
#   "biggest_risk": "...",
#   "action": "..."
# }
# """)

# print("=== Prompt ===")
# try:
#     data = json.loads(answer)
#     print(f"""
# Top Metric: {data["top_metric"]}

# Biggest Risk: {data["biggest_risk"]}

# Recommended Action: {data["action"]}
# """)
# except json.JSONDecodeError:
#     print("Failed to parse JSON. Raw output:")
#     print(answer)
