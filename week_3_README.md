# Week 3 — Agents & Memory

An exploration of memory strategies and multi-agent orchestration, building from simple in-memory conversation history up to a full research-and-report pipeline.

---

## short-term-mem.py

**What it does**
A conversational assistant that remembers the current session. Every exchange is retained so the model can refer back to earlier turns within the same run.

**How it works**
Each time the user sends a message, the full conversation history (stored in a list) is appended and passed back to the LLM on the next call.

**Key takeaway**
- Maintaining chat history as an in-memory list gives the model context across turns.

---

## long-term-mem.py

**What it does**
Extends the short-term memory approach with persistent storage so conversations survive between sessions.

**How it works**
All exchanges are written to ChromaDB with the username as a metadata key. On each new query, the relevant history is retrieved by username before calling the LLM.

**Key takeaway**
- ChromaDB as a local vector store enables LLM memory that persists beyond a single process.

---

## langchain_translation.py

**What it does**
A two-stage LangChain pipeline: first summarises an input sentence, then translates the summary into French.

**How it works**
Two independent chains are composed in sequence — each chain builds a prompt, calls the LLM, and parses the output. The summary chain's output becomes the translation chain's input.

```
user input → summary_chain → summary → translation_chain → French output
```

**Key takeaways**
- How LangChain chains work (prompt → LLM → output parser).
- Chaining two specialised pipelines so each handles one focused task.

---

## langchain_two_agent

**What it does**
A planner/executor multi-agent system. One agent breaks a goal into actionable steps; a second agent executes each step using tools, then a final summary is produced.

**How it works**
The planner agent receives the user goal and "returns a list of steps for the executor to carry out are needed. The executor agent runs each step sequentially, collects the results, and feeds them to a summarisation call.

```
user_goal → planner_agent → [step1, step2, step3, ...]
                                    ↓
                          for each step:
                              executor_agent(step, tools) → result
                                    ↓
                          collect all results
                                    ↓
                          final summary
```

**Key takeaways**
- Connecting agents to tools (functions the LLM can invoke).
- Separating concerns across agents: one plans, one executes.

---

## mini_project_w3.py

**What it does**
A research assistant that:
1. Searches the web for facts on a topic (via Tavily).
2. Stores findings in ChromaDB to avoid redundant searches.
3. Generates a structured report or answers follow-up questions directly from memory.

**How it works**
On each user request, ChromaDB is queried first. If relevant facts already exist, they are used directly. Otherwise, Tavily fetches new information, which is saved to ChromaDB before being used. The LangChain chain then either answers the question or formats a structured report depending on what the user asked.

```
user_goal → query ChromaDB ──── facts found ──→ collect facts
                  │                                    ↓
                  └── no facts → Tavily search     answer question
                                     │                  or
                                     └── save to    structured report
                                         ChromaDB
```

**Key takeaways**
- Combining all prior building blocks into one end-to-end pipeline.
- Tavily for real-time web search within an agent.
- ChromaDB as a semantic cache to skip repeat searches.
- LangChain chain for both free-form Q&A and structured report generation.


## What surprised me this week
1. LangChain's `|` pipe operator makes chaining feel suprise since very easy to use — 
   but the lambda bridge (`lambda x: {"input": x}`) is easy to forget 
   and gives a cryptic error when missing. I prefer add those into the function and call this with input

2. ChromaDB queries return results even when they are not very relevant — 
   there is no built-in "no match" response. You have to check 
   `documents == [[]]` yourself.

3. The Planner agent works better without tools — giving it tools 
   confused it into trying to execute steps itself instead of just planning.

---

## Questions I still have going into Week 4

1. How do I make the research agent search across multiple pages of a website, not just the top result snippet? 

2. Index for Chroma and ability to integrate to other db (not only splite for now)
  Any caching

3. How does LangGraph differ from my manual planner/executor pattern. it is just help for cleaner coding or any other pros?