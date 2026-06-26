# import the necessary functions
from dotenv import load_dotenv
from trafilatura import fetch_url, extract
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def summarize_url(url):
  downloaded = fetch_url(url)
  result = extract(downloaded)
  if not result:
    return {"url": url, "summary": "Could not extract content from this page."}
  summary = summary_from_page(result)
  return {
    "url": url,
    "summary": summary
  }

MAX_CHARS = 8000  # ~2000 tokens, well under Groq's 12k TPM limit

def summary_from_page(result):
  prompt = PromptTemplate(
      template="""You are a helpful agent that summarizes the provided web page content.
      Summarize in 5 bullet points for a busy professional: {result}
      Also extract: main claim / target audience / credibility score (1-10)
      """,
      input_variables=["result"]
  )
  chain = prompt | llm
  return chain.invoke({"result": result[:MAX_CHARS]}).content

TEST_URLS = [
    "https://graphql-ruby.org/",        # graphql documentation
    "https://realpython.com/python-lambda/",                  # blog/tutorial
    "https://docs.python.org/3/library/functions.html",       # docs
    "https://arxiv.org/abs/2303.08774",                       # academic
    "https://en.wikipedia.org/wiki/Large_language_model",     # reference
]

for url in TEST_URLS:
  print(f"\n=== {url} ===")
  result = summarize_url(url)
  print(result["summary"])
  
# === REFLECTION ===
# What does trafilatura do better than BeautifulSoup?
# - Extracts only the main article text, strips nav/footer/ads automatically
# - BS4 requires you to find the right selectors manually per site
# - trafilatura works on any site without custom selectors

# Which URL type worked worst?
# - Academic (arxiv) — abstract only, not full paper
# - Docs pages — sometimes navigation text bleeds in

# What would you add with more time?
# - Cache results so the same URL isn't fetched twice
# - Return structured dict with main_claim, audience, credibility as separate keys
#   instead of one big summary string — easier to use programmatically