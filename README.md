# Agentic AI Study

A hands-on exploration of agentic AI patterns — tool registries, agentic loops, NLP intent extraction, web scraping pipelines, and prompt engineering — using the Groq, OpenAI, and LangChain APIs.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (and in any week subdirectory that needs it):

```
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
```

All models use **Llama 3.3 70B** via [Groq](https://console.groq.com) (`llama-3.3-70b-versatile`) unless noted.

---

## Weeks 1–3 — Core Agent Patterns

| File | What it shows |
|---|---|
| `ai_businness_agent.py` | Tool registry + agentic loop over in-memory sales data |
| `agent.py` | Rule-based router over e-commerce CSV |
| `mini_project_agent.py` | End-to-end mini agent project |
| `mini_project_w3.py` | Research agent: web search + ChromaDB semantic cache + report generation |
| `langchain_two_agent` | Planner/executor multi-agent pattern (plan → execute → summarise) |
| `chain_of_thought.py` | Chain-of-thought prompting experiments |
| `short-term-mem.py` | Conversation memory within a session |
| `long-term-mem.py` | Persistent memory across sessions (Chroma vector store) |
| `eval_reflection.py` | Self-evaluation and reflection loop |
| `chart_eval.py` | Chart generation with LLM-driven evaluation |
| `prompt_test.py` | Prompt engineering (role, format, temperature) |
| `langchain_reflection.py` | Reflection pattern via LangChain |
| `langchain_translation.py` | LangChain translation chain |

See [`week_3_README.md`](week_3_README.md) for a day-by-day writeup of the memory and multi-agent experiments above.

### Tool Registry Pattern (`ai_businness_agent.py`)

Single source of truth — function, description, and JSON Schema live in one dict entry:

```python
TOOL_REGISTRY = {
    "revenue_by_country": {
        "function": revenue_by_country,
        "description": "...",
        "parameters": {...}   # JSON Schema
    }
}

# OpenAI tool schema derived automatically — no duplication
TOOLS = [
    {"type": "function", "function": {"name": name, "description": meta["description"], "parameters": meta["parameters"]}}
    for name, meta in TOOL_REGISTRY.items()
]
```

### Agentic Loop

```
messages = [user question]

while True:
    response = llm(messages, tools=TOOLS)
    if no tool calls → break          # LLM is done
    execute tools
    append tool results → messages    # LLM sees its own output next turn
```

---

## Week 4 — Web Research Agents

### Browser / URL Summariser (`w4_browser_agent/`)

Accepts up to 5 URLs, scrapes each with Playwright, and produces a ranked JSON summary.

```bash
python -m w4_browser_agent.pipeline
```

| File | Role |
|---|---|
| `scraper.py` | Playwright-based page fetcher |
| `summarizer.py` | LLM summarisation of scraped content |
| `pipeline.py` | Orchestrates scrape → summarise → rank |

### Company Intelligence Agent (`w4_company_intelligence_agent/`)

Takes a company website, extracts structured profile fields (name, products, target market, tech stack), and optionally compares two companies side-by-side.

```bash
python -m w4_company_intelligence_agent.pipeline
```

| File | Role |
|---|---|
| `scraper.py` | Trafilatura-based content extractor |
| `summarizer.py` | LLM-powered structured profile extraction |
| `pipeline.py` | Interactive compare flow, outputs `company_profile.json` |

Other week-4 scripts:

| File | What it shows |
|---|---|
| `w4_playwright.py` | Raw Playwright scraping experiment |
| `w4_playwright_beautifulsoup.py` | Playwright + BeautifulSoup hybrid |
| `w4_trafilatura_llm.py` | Trafilatura extraction piped to LLM |
| `w4_trafilatura_llm_ranking.py` | Adds LLM-based ranking of results |
| `llm_web_data_analysis.py` | Playwright + BeautifulSoup search-and-read pipeline, summarises top 3 results into a structured JSON report |

---

## Week 5 — NLP Intent Agent (`w5_coffee_agent/`)

Parses natural-language queries against a coffee sales CSV using LLM-based intent extraction instead of keyword routing.

```bash
python -m w5_coffee_agent.nlp
```

**Dataset columns:** `date`, `time`, `cash_type`, `card`, `price`, `coffee_name`

### How it works

```
user query
    ↓
extract_intent_and_entities()   ← LLM returns JSON {intent, entities}
    ↓
handle_query()                  ← routes to ToolRegistry
    ↓
ToolRegistry.call()             ← calls LangChain StructuredTool
```

### Intents

| Intent | Tool called | Example query |
|---|---|---|
| `top_n` | `top_n_revenue` / `top_n_item` | "what are the top sellers" |
| `filter_data` | `filter_rows` | "filter by latte" |
| `describe` | `describe_data` | "give me a breakdown of the data" |
| `plot_chart` | `plot_chart` | "make a chart" |
| `load_csv` | `load_csv` | "load sales_2025.csv" |
| `general` | direct LLM | anything else |

### Key learning

Entity key names from the LLM are unpredictable (`"column"`, `"column_name"`, `"col"`). Always use `.get()` with multiple fallback keys rather than assuming a single exact name.

### Files

| File | Role |
|---|---|
| `agent.py` | LangChain `@tool`-decorated functions over the CSV |
| `nlp.py` | ToolRegistry + LLM intent extraction + query handler |
| `analyst.py` | Higher-level analyst wrapper |
| `data/coffee_sales.csv` | Dataset |

---

## Week 5 — Classifier Router Agent (`w5/`)

A separate, simpler router: classifies a free-text query into one of four categories, then extracts structured arguments for that category's tool. Two LLM calls per query (classify → extract args) instead of one.

```bash
python -m w5.router
```

```
user query → classify_query()        ← LLM returns category name
                  ↓
        get_input_for_tool()          ← LLM extracts JSON args for that category
                  ↓
        registry.call(category, args) ← ToolRegistry dispatch
```

| File | Role |
|---|---|
| `registry.py` | `ToolRegistry` — maps category name → callable |
| `tools.py` | Calculation, date formatting, text formatting, CSV reading tools |
| `router.py` | `classify_query()` + `get_input_for_tool()` + `routing_agent()` |

**Key learning:** the two-step classify-then-extract pipeline is more accurate than a single-shot call but doubles latency/cost per query — worth it only when misrouting is expensive.

---

## RAG Evaluation Pipeline (`rag_pdf.py`, `rag_pdf_eval.py`, `rag_pdf_tunning.py`, `plot_eval.py`)

A retrieval-augmented Q&A agent over `week_3_README.md`, plus an LLM-as-judge evaluation harness used to iteratively tune chunking and retrieval parameters.

```bash
python rag_pdf.py          # ask a one-off question
python rag_pdf_tunning.py  # run the 10-question eval suite, append to eval_history.csv
python plot_eval.py        # render eval_trend.png from eval_history.csv
```

```
question → ChromaDB similarity search → top-k chunks → LLM answer (cites chunk #)
                                                              ↓
                                          LLM-as-judge scores answer 1-5 vs expected
                                                              ↓
                                          save_to_history() → eval_history.csv
                                                              ↓
                                          plot_eval.py → eval_trend.png (score over time + weakest category)
```

| File | Role |
|---|---|
| `rag_pdf.py` | `ai_answer()` — chunks `week_3_README.md` into Chroma, retrieves top-7, answers with citation + low-confidence fallback (`min_distance > 1.2` → "no relevant context") |
| `rag_pdf_eval.py` | First-pass eval: 10 fixed Q&A test cases scored by an LLM judge; includes before/after notes from tuning `chunk_overlap` and `n_results` |
| `rag_pdf_tunning.py` | Tuned version of the eval loop — appends each run's scores to `eval_history.csv` for trend tracking |
| `plot_eval.py` | Reads `eval_history.csv`, plots avg score over time + per-category breakdown to `eval_trend.png` |

**Key learnings:**
- Citation accuracy depends on chunk quality as much as retrieval quality — bad chunk boundaries (500 chars, cutting mid-concept) produce confusing citations even with correct retrieval.
- Small parameter changes compound: raising `chunk_overlap` 50→150 and `n_results` 3→5 took the eval from 8/10 to 10/10 passed.
- A low-confidence guard (`min_distance` threshold) stops the LLM from confidently answering off irrelevant chunks.
