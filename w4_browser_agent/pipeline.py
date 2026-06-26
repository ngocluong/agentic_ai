from w4_browser_agent.summarizer import summarize_url
import json
import logging
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    topic = input("What topic are you researching? ")
    
    urls = []
    while len(urls) < 5:
      url = input(f"Enter URL {len(urls) + 1}/5 (or 'quit' to stop): ")
      if url.lower() == "quit":
        break
      urls.append(url)
    print("\nProcessing URLs...")
    results = []
    for url in urls:
        start = time.time()
        print(f"  Summarizing: {url}")
        result = summarize_url(url, topic)
        results.append(result)
        logger.info(f"Processed {url} in {time.time() - start:.2f}s")
    
    # ── Rank by relevance score ───────────────────────────────────────────────────
    results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    # ── Print formatted table ─────────────────────────────────────────────────────
    print(f"\n=== Results Ranked by Relevance to: '{topic}' ===\n")
    print(f"{'#':<4} {'Rel':<5} {'Cred':<6} {'Title':<40} {'URL'}")
    print("-" * 100)
    for i, r in enumerate(results, 1):
        title = str(r.get("title", "N/A"))[:38]
        url   = str(r.get("url",   "N/A"))[:45]
        rel   = r.get("relevance_score",   0)
        cred  = r.get("credibility_score", 0)
        print(f"{i:<4} {rel:<5} {cred:<6} {title:<40} {url}")

    # ── Save JSON ─────────────────────────────────────────────────────────────────
    with open("w4_sum_ranking.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to w4_sum_ranking.json")