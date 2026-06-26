import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from w4_browser_agent import config
from w4_browser_agent.scraper import scrape_url


load_dotenv()

llm = ChatGroq(
    model= config.LLM_MODEL,
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

def summarize_url(url, topic):
    """Fetch a URL and summarize its content with relevance scoring.
    
    Args:
        url: The URL to fetch and summarize
        topic: Research topic to score relevance against
    Returns:
        Dict with keys: url, title, main_claim, credibility_score, relevance_score, summary
    """
    result = scrape_url(url)
    if not result:
        return {
            "url": url,
            "title": "N/A",
            "main_claim": "N/A",
            "credibility_score": 0,
            "relevance_score": 0,
            "summary": "Could not extract content from this page."
        }
    summary = summarize_text(result, topic)
    return {
        "url": url,
        "title": summary.get("title", "N/A"),
        "main_claim": summary.get("main_claim", "N/A"),
        "credibility_score": summary.get("credibility_score", 0),
        "relevance_score": summary.get("relevance_score", 0),
        "summary": summary.get("summary", "N/A")
    }
    
def summarize_text(result, topic):
    """Summarize text and score relevance to a topic using LLM.
    
    Args:
        text: The text content to summarize
        topic: Research topic to score relevance against
    Returns:
        Dict with keys: title, main_claim, summary, credibility_score, relevance_score
    """
    prompt = PromptTemplate(
        template="""Summarize this web page in 5 bullet points for a busy professional: {result}

        Extract: title, main claim, credibility score (1-10), relevance score to "{topic}" (1-10).

        Return JSON with keys: summary, title, main_claim, credibility_score, relevance_score
        """,
        input_variables=["result", "topic"]
    )
    chain = prompt | llm
    content = chain.invoke({"result": result[:config.MAX_CHARS], "topic": topic}).content
    return json.loads(content)  # always valid JSON — no regex needed
