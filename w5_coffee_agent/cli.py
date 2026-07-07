import json
import uuid
import time
import logging
from collections import Counter
from w5_coffee_agent.nlp import extract_intent_and_entities, handle_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
      
def run_session():
    session_id     = str(uuid.uuid4())
    total_duration = 0.0
    success        = 0
    intents        = []
    query_count = 0
    ss_history = []
    
    while True:
        query = input("May I help you: (please enter save for saving report and quit/exit/q to quit): ")
        if query.lower() in ["quit", "exit", "q"]:
          print("saving session and quit")
          if query_count == 0:
            return {
              "session_id":        session_id,
              "avg_response_time": 0,
              "success_rate":      0,
              "most_used_tool":    "",
            }
          else:
            counter = Counter(intents)
            return {
              "session_id":        session_id,
              "avg_response_time": total_duration / (query_count),
              "success_rate":      success / query_count,
              "most_used_tool":    counter.most_common(1)[0][0],
            }
        elif query.lower() == 'save':
          intent = "save_report"
          entities = {"session_history": ss_history}
          result = handle_query(intent, entities, query)
        else:
          start = time.time()
          try:
            extracted = extract_intent_and_entities(query)
            intent    = extracted["intent"]
            entities  = extracted["entities"]
            print(f"Intent:   {intent}")
            print(f"Entities: {entities}")
            if intent == "save_report":
              entities["session_history"] = ss_history
              
            result = handle_query(intent, entities, query)
            print(f"Result:   {result}\n")
            intents.append(intent)
            total_duration += time.time() - start
            is_error = isinstance(result, dict) and result.get("success") is False
            ss_history.append({ "query": query, "intent": intent, "result": result })
            if not is_error:
              success += 1
            
            query_count += 1
          except Exception as e:
            print(f"Error: {e}")
            continue

print("\n=== Coffee Agent ===\n")
stats  = run_session()
logger.info("Session stats: %s", stats)
with open("w5_cli.json", "a") as f:
  json.dump(stats, f)
  f.write("\n")