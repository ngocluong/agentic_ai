from openai import OpenAI
import os
import json
from prompt_toolkit import prompt
import time
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

sales_data = [
    {
        "date": "2025-01",
        "country": "USA",
        "category": "Electronics",
        "segment": "Enterprise",
        "revenue": 120000,
        "customers": 120,
        "orders": 300,
    },
    {
        "date": "2025-01",
        "country": "USA",
        "category": "Fashion",
        "segment": "SMB",
        "revenue": 40000,
        "customers": 80,
        "orders": 140,
    },
    {
        "date": "2025-01",
        "country": "Canada",
        "category": "Electronics",
        "segment": "Enterprise",
        "revenue": 70000,
        "customers": 60,
        "orders": 150,
    },
    {
        "date": "2025-02",
        "country": "USA",
        "category": "Electronics",
        "segment": "Enterprise",
        "revenue": 150000,
        "customers": 140,
        "orders": 350,
    },
    {
        "date": "2025-02",
        "country": "Canada",
        "category": "Fashion",
        "segment": "SMB",
        "revenue": 50000,
        "customers": 90,
        "orders": 180,
    },
    {
        "date": "2025-02",
        "country": "Germany",
        "category": "Electronics",
        "segment": "Enterprise",
        "revenue": 90000,
        "customers": 75,
        "orders": 200,
    },
    {
        "date": "2025-03",
        "country": "USA",
        "category": "Health",
        "segment": "Consumer",
        "revenue": 60000,
        "customers": 150,
        "orders": 250,
    },
    {
        "date": "2025-03",
        "country": "Germany",
        "category": "Fashion",
        "segment": "SMB",
        "revenue": 45000,
        "customers": 100,
        "orders": 160,
    },
    {
        "date": "2025-03",
        "country": "Canada",
        "category": "Health",
        "segment": "Consumer",
        "revenue": 55000,
        "customers": 130,
        "orders": 210,
    },
]
MAX_ITERATIONS = 10

def revenue_by_country():
    """Return total revenue aggregated by country."""
    country_revenue = {}
    for record in sales_data:
        country = record["country"]
        revenue = record["revenue"]
        if country not in country_revenue:
            country_revenue[country] = 0
        country_revenue[country] += revenue
    return country_revenue


def revenue_by_country_name(name):
    """Return total revenue for a specific country given its name."""
    country_revenue = revenue_by_country()
    return {name: country_revenue.get(name, 0)}


def top_category():
    category_revenue = {}
    for record in sales_data:
        category = record["category"]
        revenue = record["revenue"]
        if category not in category_revenue:
            category_revenue[category] = 0
        category_revenue[category] += revenue
    top_cat = max(category_revenue, key=category_revenue.get)
    return {"top_category": top_cat, "revenue": category_revenue[top_cat]}


def revenue_by_segment():
    """Return total revenue aggregated by customer segment."""
    segment_revenue = {}
    for record in sales_data:
        segment = record["segment"]
        revenue = record["revenue"]
        if segment not in segment_revenue:
            segment_revenue[segment] = 0
        segment_revenue[segment] += revenue
    return segment_revenue

TOOL_REGISTRY = {
    "revenue_by_country": {
        "function": revenue_by_country,
        "description": "Calculate total revenue grouped by country",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "revenue_by_country_name": {
        "function": revenue_by_country_name,
        "description": "Calculate total revenue for a specific country",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name of the country"}
            },
            "required": ["name"],
        },
    },
    "revenue_by_segment": {
        "function": revenue_by_segment,
        "description": "Calculate total revenue grouped by customer segment",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "top_category": {
        "function": top_category,
        "description": "Find highest revenue category",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# Derived from TOOL_REGISTRY — single source of truth
TOOLS = [
    {
        "type": "function",
        "function": {"name": name, "description": meta["description"], "parameters": meta["parameters"]},
    }
    for name, meta in TOOL_REGISTRY.items()
]

def build_prompt(question, options=None):
    options = options or {}
    role = options.get("role", "data analyst")
    return f"""
  You are a {role}. 
  You MUST use tools to answer questions.
  Never calculate manually.
  Question: {question}\n\n
  Use the appropriate tool to answer the question.
  Return your answer in a clear and concise manner.
  the answer should be in a structured format, valid JSON with keys.
  """

def validate_result(result):
  if result is None:
      return False

  if isinstance(result, dict) and len(result) == 0:
      return False

  return True

def chart_agent(question, output_file="chart1.png"):
    messages = [
        {
            "role": "system",
            "content": """You are a senior data analyst. 
            Always use tools when needed.
            """
        },
        {
            "role": "user",
            "content": (
                f"""{question}\n\n
                Return ONLY a python code
                No explanation, no markdown, Python code string using matplotlib that:
                Creates the chart
                Saves it as '{output_file}'
                """
            ),
        },
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    message = response.choices[0].message
    raw = message.content

    try:
        result = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        result = raw

    return result


def agent(question):
    messages = [
        {
            "role": "system",
            "content": """You are a senior data analyst. 
            Always use tools when needed.
            """
        },
        {
            "role": "user",
            "content": (
                f"{question}\n\n"
                "Return ONLY a JSON object. Use the exact keys returned by the tool — do NOT rename or reformat them. "
                "For country revenue, the key must be the country name (e.g. {\"USA\": 370000}). "
                "For segment revenue, each key must be the segment name. "
                "For top category, use exactly {\"top_category\": ..., \"revenue\": ...}. "
                "No explanation, no markdown, just the raw JSON."
            ),
        },
    ]

    execution_trace = []
    i = 0

    while i < MAX_ITERATIONS:
        i += 1

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name

            try:
                raw_args = tool_call.function.arguments
                function_args = json.loads(raw_args) if raw_args else {}
                if function_args is None:
                    function_args = {}
            except json.JSONDecodeError as e:
                execution_trace.append({
                    "step": len(execution_trace) + 1,
                    "tool_name": function_name,
                    "status": "json_parse_failed",
                    "error": str(e),
                })
                continue

            func = TOOL_REGISTRY.get(function_name, {}).get("function")

            try:
                start = time.time()
                result = func(**function_args)
                duration = time.time() - start

                execution_trace.append({
                    "step": len(execution_trace) + 1,
                    "tool_name": function_name,
                    "arguments": function_args,
                    "result": result,
                    "status": "success",
                    "duration": round(duration, 4),
                })

            except Exception as e:
                result = {"error": str(e)}

                execution_trace.append({
                    "step": len(execution_trace) + 1,
                    "tool_name": function_name,
                    "status": "failed",
                    "error": str(e),
                })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    return {
        "question": question,
        "steps": execution_trace,
        "final_answer": "Max iterations reached without final answer",
    }

# question = input("Ask: ")
# prompt = build_prompt(question, options={"role": "senior data analyst"})
# print(agent(prompt))

# print(revenue_by_segment())
