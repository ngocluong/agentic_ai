from ai_businness_agent import chart_agent, sales_data
import difflib

question = "Make a bar chart of monthly revenue for each country."
raw = chart_agent(question)

try:
  exec(raw) if isinstance(raw, str) else raw
except Exception as e:
  print(f"Failed to parse the result as Python code. Raw output: {raw}, error: {e}")

# Reflect on the code and improve it
reflection = chart_agent(
    f"""
    Here is the original chart code: {raw}
    Here is the actual data to use: {sales_data}
    
    Improve the code to be more professional.
    Consider: title, axis labels, colors, grid, legend, value labels on bars.
    Keep the same data — do not replace it with fake data.
    Save as chart_reflection.png
    """
)

try:
  exec(reflection) if isinstance(reflection, str) else reflection
except Exception as e:
  print(f"Failed to parse the result as Python code. Raw output: {reflection}, error: {e}")
  
  
diff = difflib.unified_diff(
  raw.splitlines(),
  reflection.splitlines(),
  lineterm='',
  fromfile='chart_v1',
  tofile='chart_v2'
)
print('\n'.join(diff))