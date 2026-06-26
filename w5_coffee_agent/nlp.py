import json

from langchain_groq import ChatGroq
from w5_coffee_agent.agent import load_csv, describe_data, filter_rows, top_n_revenue, top_n_item, plot_chart

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name: str, func):
        """Register a tool by name."""
        self._tools[name] = func

    def list_tools(self) -> list:
        """Return list of available tool names."""
        return list(self._tools.keys())

    def call(self, name: str, inputs: dict) -> dict:
        """Call a tool by name with inputs."""
        if name not in self._tools:
            return {"success": False, "result": None, "error": f"Unknown tool: {name}"}
        return self._tools[name].func(**inputs)

registry = ToolRegistry()
registry.register("load_csv", load_csv)
registry.register("describe_data", describe_data)
registry.register("filter_rows", filter_rows)
registry.register("top_n_revenue", top_n_revenue)
registry.register("top_n_item", top_n_item)
registry.register("plot_chart", plot_chart)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

def extract_intent_and_entities(query: str) -> dict:
    messages = [
      ("system", """Extract the intent and entities from this query.
        Available columns in the dataset: date, time, cash_type, card, price, coffee_name
        Intent must be one of: top_n, filter_data, describe, plot_chart, load_csv, general
        Entities are specific values mentioned (numbers, names, column names, dates, chart types)

        Return JSON only:
          {
            "intent": "top_n",
            "entities": {
              "n": 5,
              "metric": "revenue"
            }
          }
      """),
      ("human", query),
    ]
    return llm.invoke(messages)

def handle_query(query: str) -> dict:
    extracted = json.loads(extract_intent_and_entities(query).content)
    
    intent   = extracted["intent"]
    entities = extracted["entities"]
    
    if intent == "top_n":
        n      = entities.get("n") or 5
        metric = entities.get("metric", "revenue")
        if metric == "revenue":
            return registry.call("top_n_revenue", {"n": n})
        else:
            return registry.call("top_n_item", {"n": n})
    
    elif intent == "filter_data":
        return registry.call("filter_rows", {
            "column": entities.get("column") or entities.get("column_name", "coffee_name"),
            "value":  entities.get("value", "")
        })
    
    elif intent == "describe":
        return registry.call("describe_data", {
            "column": entities.get("column", "")
        })
    
    elif intent == "plot_chart":
        return registry.call("plot_chart", {
            "column":     entities.get("column", "coffee_name"),
            "chart_type": entities.get("chart_type", "bar")
        })
    
    elif intent == "load_csv":
        return registry.call("load_csv", {
            "filepath": entities.get("filepath", "")
        })
    else:
        return {"success": True, "result": llm.invoke(query).content, "error": None}

TEST_QUERIES = [
    "show me last month numbers",
    "what are the top sellers",
    "filter by latte",
    "give me a breakdown of the data",
    "make a chart",
]

print("\n=== Testing Ambiguous Queries ===\n")
for query in TEST_QUERIES:
    print(f"Query: {query}")
    extracted = json.loads(extract_intent_and_entities(query).content)
    print(f"Intent: {extracted['intent']}")
    print(f"Entities: {extracted['entities']}")
    result = handle_query(query)
    print(f"Result: {result}")
    print()

# === REFLECTION ===
# What worked well:
# - top_n, describe, plot_chart all handled ambiguous queries correctly
# - Default fallbacks (n=5, column="coffee_name") saved 2 queries from failing

# What failed:
# - "filter by latte" → LLM used "column_name" key instead of "column"
#   Fix: check both keys, or add column names to the prompt context

# - "show me last month numbers" → no date filtering tool available
#   Fix: add a filter_by_date tool, or handle time_period entity specially

# Key learning:
# Entity key names are unpredictable — LLM may return "column_name", "col", "field"
# Always use .get() with multiple fallback keys, never assume exact key names

# When does NLP extraction beat simple routing (Day 33)?
# - When queries are natural language with specific values embedded
# - "top 3" vs "top 5" — router can't distinguish, NLP extracts the number
# - "filter by latte" — router routes to csv, NLP extracts the actual value