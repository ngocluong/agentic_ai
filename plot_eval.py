import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def chart_agent(output_file="chart1.png"):
  messages = [
      {
          "role": "system",
          "content": """You are a senior data analyst. """
      },
      {
          "role": "user",
          "content": (
              f"""
              Read file 'eval_history.csv' and create a chart.
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
  )

  message = response.choices[0].message
  raw = message.content

  try:
      result = json.loads(raw) if isinstance(raw, str) else raw
  except (json.JSONDecodeError, TypeError):
      result = raw

  return result

def strip_code_fence(code):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
    return match.group(1) if match else code

reflection = chart_agent()
print("Reflection code:", reflection)
if isinstance(reflection, str):
    exec(strip_code_fence(reflection))