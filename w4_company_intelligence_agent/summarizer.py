import json
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from w4_company_intelligence_agent import config
from w4_company_intelligence_agent.scraper import scrape_url

load_dotenv()

llm = ChatGroq(
    model= config.LLM_MODEL,
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

def get_summary(url):
    """Fetch a URL and summarize its content with relevance scoring.
    
    Args:
        url: The URL to fetch and summarize
    Returns:
        Dict with keys: company name, products, target market, tech stack, summary
    """
    
    print(f"Fetching and summarizing content from: {url}")
    result = get_relevance(url)
    if not result:
        return {
            "url": url,
            "company_name": "N/A",
            "products": "N/A",
            "target_market": "N/A",
            "tech_stack": "N/A",
            "summary": "Could not extract content from this page."
        }
    about_result = scrape_url(result.get("about_page_url", "N/A"))
    products_result = scrape_url(result.get("products_page_url", "N/A"))
    return summarize_text(result.get("content", "N/A"), about_result, products_result)

def get_relevance(url):
    """Fetch a URL and about page URL and Products URL.
    
    Args:
        url: The URL to fetch and summarize
    Returns:
        Dict with keys: url, content, about_page_url, products_page_url
    """
    result = scrape_url(url)
    if not result:
        return {
            "url": url,
            "content": "N/A",
            "about_page_url": "N/A",
            "products_page_url": "N/A"
        }
    summary_url = extract_about_and_products(result)
    base = "{0.scheme}://{0.netloc}".format(urlparse(url))

    def resolve(u):
        if not u or u == "N/A":
            return "N/A"
        return urljoin(base, u)

    print(f"Extracted about page URL: {resolve(summary_url.get('about_page_url', 'N/A'))}")
    print(f"Extracted products page URL: {resolve(summary_url.get('products_page_url', 'N/A'))}")
    return {
        "url": url,
        "content": result,
        "about_page_url": resolve(summary_url.get("about_page_url", "N/A")),
        "products_page_url": resolve(summary_url.get("products_page_url", "N/A"))
    }

def extract_about_and_products(result):
    """Extract about page URL and products page URL from text using LLM.
    
    Args:
        text: The text content to extract from
    Returns:
        Dict with keys: about_page_url, products_page_url
    """
    prompt = PromptTemplate(
        template="""Extract the about page URL and products page URL from this web page content: {result}

        Return JSON with keys: about_page_url, products_page_url
        """,
        input_variables=["result"]
    )
    chain = prompt | llm
    content = chain.invoke({"result": result[:config.MAX_CHARS]}).content
    return json.loads(content)  # always valid JSON — no regex needed

def summarize_text(home_page_content, about_result, products_result):
    """Summarize text and extract company information using LLM.
    
    Args:
        home_page_content: The text content from the home page
        about_result: The text content from the about page
        products_result: The text content from the products page
    Returns:
        Dict with keys: company_name, products, target_market, tech_stack, summary
    """
    prompt = PromptTemplate(
        template="""Summarize this company's web page content: {home_page_content}
        additionally, use the about page content: {about_result} 
        and products page content: {products_result} to extract relevant information.
        Extract: company name, products, target market, tech stack and summary
        Return JSON with keys: company_name, products, target_market, tech_stack, summary
        """,
        input_variables=["home_page_content", "about_result", "products_result"]
    )
    chain = prompt | llm
    content = chain.invoke({
        "home_page_content": home_page_content[:config.MAX_CHARS],
        "about_result": about_result[:config.MAX_CHARS],
        "products_result": products_result[:config.MAX_CHARS]
    }).content
    return json.loads(content)  # always valid JSON — no regex needed
