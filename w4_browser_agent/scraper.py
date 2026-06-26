from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
from bs4 import BeautifulSoup
from w4_browser_agent import config

def scrape_url(url: str) -> str:
    """Fetch and extract main text content from a URL.
    
    Args:
        url: The URL to scrape
    Returns:
        Extracted text content, or empty string if failed
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=config.TIMEOUT)
            except PlaywrightTimeoutError:
                pass
            try:
                html_content = page.content()
            except PlaywrightError:
                return ""
        finally:
            browser.close()
        soup = BeautifulSoup(html_content, "lxml")
        main_content = soup.find('main')
        if main_content:
            return main_content.get_text(separator="\n", strip=True)
        else:
            return soup.get_text(separator="\n", strip=True)
