# Agentic AI Study

A hands-on exploration of agentic AI patterns — tool registries, agentic loops, and prompt engineering — using the Groq and OpenAI APIs.

## Key Concepts

### Tool Registry (`ai_business_agent.py`)

The main agent demonstrates the **single source of truth** registry pattern:

```python
TOOL_REGISTRY = {
    "revenue_by_country": {
        "function": revenue_by_country,   # execution target
        "description": "...",             # passed to the LLM
        "parameters": {...}               # JSON Schema
    }
}

# OpenAI tool schema derived automatically — no duplication
TOOLS = [
    {"type": "function", "function": {"name": name, "description": meta["description"], "parameters": meta["parameters"]}}
    for name, meta in TOOL_REGISTRY.items()
]
```

Adding a new tool only requires one entry in `TOOL_REGISTRY`.

### Agentic Loop

```
messages = [user question]

while True:
    response = llm(messages, tools=TOOLS)

    if no tool calls → break          # LLM is done; final answer is here

    execute tools
    append tool results → messages    # LLM sees its own tool output next turn
```

The loop continues until the LLM stops requesting tools, allowing multi-step reasoning without a hardcoded number of passes.

## Files

| File | Purpose |
|---|---|
| `ai_business_agent.py` | Main agent — tool registry + agentic loop over in-memory sales data |
| `agent.py` | Simple rule-based agent over e-commerce CSV |
| `prompt_test.py` | Prompt engineering experiments (role, format, temperature) |
| `test.py` | Groq API smoke test |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key   # only needed for prompt_test.py
```

## Running the main agent

```bash
python ai_business_agent.py
```

Example questions:
- `What is the revenue by country?`
- `Which country has the highest revenue?`
- `What is the top category?`
- `Break down revenue by customer segment`

## Available Tools

| Tool | Description |
|---|---|
| `revenue_by_country` | Total revenue grouped by country |
| `revenue_by_country_name` | Revenue for a specific country |
| `revenue_by_segment` | Revenue grouped by customer segment (Enterprise / SMB / Consumer) |
| `top_category` | Highest-revenue product category |

## Model

Uses **Llama 3.3 70B** via [Groq](https://console.groq.com) (`llama-3.3-70b-versatile`).
