TEST_CASES = [
  # ── what_is_rag.txt ──────────────────────────────────────────────────────
  {
      "question": "What are the two phases of RAG and when does each happen?",
      "expected": "The indexing phase happens once — documents are chunked, embedded, and stored in a vector database. The query phase happens on every question — the question is embedded, similar chunks are retrieved, and injected into the LLM prompt."
  },
  {
      "question": "Why does RAG reduce hallucination compared to plain prompting?",
      "expected": "RAG grounds the answer in real documents rather than relying on the LLM's training data, and allows updating the knowledge base without retraining the model."
  },

  # ── what_is_langchain.txt ────────────────────────────────────────────────
  {
      "question": "What does StrOutputParser return compared to JsonOutputParser?",
      "expected": "StrOutputParser returns a plain string. JsonOutputParser returns a Python dictionary."
  },
  {
      "question": "What is LCEL and when does a chain execute?",
      "expected": "LCEL is LangChain Expression Language. The chain is lazy — it does not execute until invoke is called."
  },

  # ── what_is_chromadb.txt ─────────────────────────────────────────────────
  {
      "question": "What is the difference between PersistentClient and EphemeralClient in ChromaDB?",
      "expected": "PersistentClient saves data to disk and survives script restarts. EphemeralClient keeps data only in memory and is lost when the process ends."
  },
  {
      "question": "What does a ChromaDB distance above 1.2 indicate?",
      "expected": "A distance above 1.2 indicates a poor match between the query and the retrieved chunk."
  },
  # ── what_is_react_agent.txt ──────────────────────────────────────────────────
  {
      "question": "What are the three steps in the ReAct loop?",
      "expected": "Think — the LLM reasons and decides which tool to call. Act — the agent calls the chosen tool. Observe — the agent reads the tool result and decides the next step."
  },
  {
      "question": "What is the purpose of max_steps in a ReAct agent?",
      "expected": "max_steps sets a hard limit on the number of think-act-observe cycles to prevent the agent looping indefinitely. When reached, the agent stops and returns whatever it has."
  },

  # ── what_is_langgraph.txt ────────────────────────────────────────────────────
  {
      "question": "What is the difference between a node and an edge in LangGraph?",
      "expected": "A node is a Python function that processes state and returns updates. An edge connects two nodes and controls which node runs next."
  },
  {
      "question": "What does MemorySaver do in LangGraph?",
      "expected": "MemorySaver is a checkpointer that saves the complete state after each node runs, allowing the agent to resume from where it left off and enabling multi-turn conversations."
  },
]

