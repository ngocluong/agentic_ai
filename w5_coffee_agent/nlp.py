import json
import uuid
import time
import logging
from collections import Counter

from langchain_groq import ChatGroq
from w5_coffee_agent.agent import save_report, load_csv, describe_data, filter_rows, top_n_revenue, top_n_item, plot_chart

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Extract the intent and entities from this query.
Available columns in the dataset: date, time, cash_type, card, price, coffee_name
Intent must be one of: top_n, filter_data, describe, plot_chart, save_report, load_csv, general
Entities are specific values mentioned (numbers, names, column names, dates, chart types)

Return JSON only:
  {
    "intent": "top_n",
    "entities": {
      "n": 5,
      "metric": "revenue"
    }
  }
"""


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name: str, func):
        self._tools[name] = func

    def list_tools(self) -> list:
        return list(self._tools.keys())

    def call(self, name: str, inputs: dict):
        if name not in self._tools:
            return {"success": False, "result": None, "error": f"Unknown tool: {name}"}
        return self._tools[name].func(**inputs)


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)

registry = ToolRegistry()
for _name, _func in [
    ("load_csv",      load_csv),
    ("describe_data", describe_data),
    ("filter_rows",   filter_rows),
    ("top_n_revenue", top_n_revenue),
    ("top_n_item",    top_n_item),
    ("plot_chart",    plot_chart),
    ("save_report",    save_report),
]:
    registry.register(_name, _func)


def extract_intent_and_entities(query: str) -> dict:
    response = llm.invoke([("system", _SYSTEM_PROMPT), ("human", query)])
    return json.loads(response.content)


def handle_query(intent: str, entities: dict, query: str):
    if intent == "top_n":
        n      = entities.get("n") or 5
        metric = entities.get("metric", "revenue")
        tool   = "top_n_revenue" if metric == "revenue" else "top_n_item"
        return registry.call(tool, {"n": n})

    if intent == "filter_data":
        # LLM sometimes returns the column name as the key itself
        # e.g. {"coffee_name": "Latte"} instead of {"column": "coffee_name", "value": "Latte"}
        known_columns = ["coffee_name", "cash_type", "date", "card"]
        column = entities.get("column") or entities.get("column_name", "coffee_name")
        value  = entities.get("value", "")
        
        # handle case where LLM used column name as key directly
        for col in known_columns:
            if col in entities and col not in ["column", "column_name"]:
                column = col
                value  = entities[col]
                break
        
        return registry.call("filter_rows", {"column": column, "value": value})

    if intent == "describe":
        return registry.call("describe_data", {"column": entities.get("column", "")})

    if intent == "plot_chart":
        return registry.call("plot_chart", {
            "column":     entities.get("column", "coffee_name"),
            "chart_type": entities.get("chart_type", "bar"),
        })
    
    if intent == "save_report":
        return registry.call("save_report", {
            "history": entities.get("session_history", []),
            "file_name": entities.get("filename", "report.txt"),
        })

    if intent == "load_csv":
        import os
        filepath = entities.get("filepath") or entities.get("file_path", "")
        # resolve relative to the package data folder
        if not os.path.isabs(filepath):
            package_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(package_dir, filepath)
        return registry.call("load_csv", {"filepath": filepath})

    response = llm.invoke([
        ("system", "Answer the user's question. Return JSON with a single 'result' key."),
        ("human", query),
    ])
    return {"success": True, "result": json.loads(response.content).get("result"), "error": None}


def run_session(max_queries: int = 5) -> dict:
    session_id     = str(uuid.uuid4())
    total_duration = 0.0
    success        = 0
    intents        = []

    for _ in range(max_queries):
        query = input("May I help you: ")
        start = time.time()

        extracted = extract_intent_and_entities(query)
        intent    = extracted["intent"]
        entities  = extracted["entities"]

        print(f"Intent:   {intent}")
        print(f"Entities: {entities}")

        result = handle_query(intent, entities, query)
        print(f"Result:   {result}\n")

        intents.append(intent)
        total_duration += time.time() - start
        is_error = isinstance(result, dict) and result.get("success") is False
        if not is_error:
            success += 1

    counter = Counter(intents)
    return {
        "session_id":        session_id,
        "avg_response_time": total_duration / max_queries,
        "success_rate":      success / max_queries,
        "most_used_tool":    counter.most_common(1)[0][0],
    }


if __name__ == "__main__":
    print("\n=== Coffee Agent ===\n")
    stats = run_session()
    logger.info("Session stats: %s", stats)
    with open("w5_nlp.json", "a") as f:
        json.dump(stats, f)
        f.write("\n")


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

# Additions:
# Logging: session_id + stats make debugging across runs possible
# Counter: clean way to find most_used_tool without manual counting
# JSONL format: one JSON object per line — append-friendly, grep-friendly
# avg_response_time 0.41s — mostly LLM latency, not Python overhead
# success_rate 1.0 — fallback defaults did their job (n=5 when not specified)