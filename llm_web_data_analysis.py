from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def analysis_web(query: str):
  summaries = research_and_summarize(query)
  report = generate_report(query, summaries)
  output = {
        "question": query,
        "summaries": summaries,
        "report": report
    }
  with open("analysis_report.json", "w") as f:
    json.dump(output, f, indent=2)
    
  print("Saved to analysis_report.json")
  return report

def research_and_summarize(query: str):
  with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://duckduckgo.com/", wait_until="networkidle")
    page.locator('input[name="q"]').fill(query)
    page.locator('input[name="q"]').press("Enter")
    page.wait_for_selector('[data-testid="result"]', timeout=15000)
    results = page.locator('[data-testid="result"]').all()
    summaries = []
    for result in results:
      if len(summaries) >= 3:
        break
      if result.locator('[data-testid="result-extras-url-link"]').count() > 0:
        link = result.locator('[data-testid="result-extras-url-link"]').first
        link.click()  
        try:
          page.wait_for_selector('h1', timeout=10000)
        except Exception:
          pass  # continue even if no h1
        html_content = page.content()
        soup = BeautifulSoup(html_content, "lxml")
        summaries.append(summary_from_page(soup))
        page.go_back()
        page.wait_for_load_state("networkidle")
    browser.close()
    return summaries
  
def summary_from_page(soup):
  text = soup.get_text()[:4000]
  prompt = PromptTemplate(
      template="""You are a helpful agent that summarizes the provided web page content.
      Summarize the following facts in 2 sentences: {text}""",
      input_variables=["text"]
  )
  chain = prompt | llm
  return chain.invoke({"text": text}).content

def generate_report(question, facts):
    prompt = PromptTemplate(
      template=
        """Based on these facts: {facts}
        Generate a structured JSON report about: {question}

        Format exactly as:
        ## Summary
        (2 sentences)

        ## What is the common theme
        (bullet points)""",
      input_variables=["question", "facts"]
    )
    chain = prompt | llm
    return chain.invoke({"question": question, "facts": facts}).content

question = input("What Can I help: ")
print(analysis_web(question))


# === REFLECTION ===
# What does this pipeline do that Tavily (Day 23) doesn't?
# - Actually visits and reads full page content, not just snippets
# - More tokens but richer context for summarization

# What would break this?
# - Pages that require login or JavaScript to render content
# - Sites that block headless browsers (Cloudflare, etc.)
# - go_back() fails if the clicked page opens in a new tab

# When would you use Playwright+BS4 vs Tavily?
# - Tavily: fast, reliable, good for structured search results
# - Playwright+BS4: when you need full page content or need to interact with the page

# What's the weakest part of this code?
# - soup.get_text()[:4000] cuts mid-sentence — could miss key info
# - No deduplication — if 2 articles cover same content, summaries repeat