import json
from dotenv import load_dotenv
from trafilatura import fetch_url, extract
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

MAX_CHARS = 8000

def summarize_url(url, topic):
    downloaded = fetch_url(url)
    result = extract(downloaded)
    if not result:
        return {
            "url": url,
            "title": "N/A",
            "main_claim": "N/A",
            "credibility_score": 0,
            "relevance_score": 0,
            "summary": "Could not extract content from this page."
        }
    summary = summary_from_page(result, topic)
    return {
        "url": url,
        "title": summary.get("title", "N/A"),
        "main_claim": summary.get("main_claim", "N/A"),
        "credibility_score": summary.get("credibility_score", 0),
        "relevance_score": summary.get("relevance_score", 0),
        "summary": summary.get("summary", "N/A")
    }

def summary_from_page(result, topic):
    prompt = PromptTemplate(
        template="""Summarize this web page in 5 bullet points for a busy professional: {result}

        Extract: title, main claim, credibility score (1-10), relevance score to "{topic}" (1-10).

        Return JSON with keys: summary, title, main_claim, credibility_score, relevance_score
        """,
        input_variables=["result", "topic"]
    )
    chain = prompt | llm
    content = chain.invoke({"result": result[:MAX_CHARS], "topic": topic}).content
    return json.loads(content)  # always valid JSON — no regex needed

# ── Input ─────────────────────────────────────────────────────────────────────
topic = input("What topic are you researching? ")

urls = []
while len(urls) < 5:
    url = input(f"Enter URL {len(urls) + 1}/5 (or 'quit' to stop): ")
    if url.lower() == "quit":
        break
    urls.append(url)

# ── Process ───────────────────────────────────────────────────────────────────
print("\nProcessing URLs...")
results = []
for url in urls:
    print(f"  Summarizing: {url}")
    result = summarize_url(url, topic)
    results.append(result)

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
with open("sum_ranking.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results to sum_ranking.json")