from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

MAX_RETRIES = 3

def search_amazon(query: str) -> list[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _run_scrape(query)
        except PlaywrightTimeoutError as e:
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Timeout: {e}")
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt + random.uniform(1, 3)
                print(f"Waiting {wait:.1f}s before retry...")
                time.sleep(wait)
        except Exception as e:
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Error: {e}")
            if attempt == MAX_RETRIES:
                raise
    return []

def _run_scrape(query: str) -> list[dict]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
        try:
            page.goto("https://www.amazon.com/", wait_until="domcontentloaded")

            # Detect CAPTCHA / bot-check page
            if "robot" in page.title().lower() or "captcha" in page.content().lower():
                raise Exception("Amazon rate limit / CAPTCHA detected — try again later or use a different IP")

            search = page.locator('input[id="twotabsearchtextbox"]')
            search.wait_for(state="visible", timeout=15000)
            search.click()
            search.fill(query)
            search.press("Enter")

            page.wait_for_selector('[data-component-type="s-search-result"]', timeout=20000)
            page.wait_for_load_state("domcontentloaded")
            html_content = page.content()
        finally:
            browser.close()

    soup = BeautifulSoup(html_content, "lxml")
    results = []
    for item in soup.find_all(attrs={"data-component-type": "s-search-result"}):
        try:
            h2 = item.find("h2")
            title = h2.get_text(strip=True) if h2 else "N/A"

            # Link is a sibling <a> of <h2>, not nested inside it
            title_section = item.find("div", attrs={"data-cy": "title-recipe"})
            link_tag = title_section.find("a") if title_section else None
            if not link_tag:
                link_tag = item.find("a", class_=lambda c: c and "a-link-normal" in c and "s-link-style" in c)
            link = "https://www.amazon.com" + link_tag["href"] if link_tag and link_tag.get("href") else "N/A"

            price_el = item.find("span", class_="a-offscreen")
            price = price_el.text.strip() if price_el else "N/A"

            results.append({"title": title, "price": price, "link": link})
            if len(results) == 10:
                break
        except Exception as e:
            print(f"Skipping item due to parse error: {e}")
            continue

    return results


results = search_amazon("Hoka shoes")

if results:
    for r in results:
        print(f"Title: {r['title']}\nPrice: {r['price']}\nLink: {r['link']}\n")
    df = pd.DataFrame(results)
    df.to_csv('w4_playwright_beautifulsoup.csv', index=False)
    print(f"Saved {len(results)} results to CSV.")
else:
    print("No results found.")


# === REFLECTION ===
# Why Amazon instead of HackerNews?
# - Amazon is harder — dynamic content, anti-bot, CAPTCHA
# - HackerNews is static HTML — simpler but less realistic

# What was the hardest part?
# - Finding the right CSS selectors for title/price/link
# - Amazon changes its HTML structure frequently

# What would break this scraper?
# - Amazon updates their HTML class names (happens often)
# - IP gets blocked after too many requests
# - CAPTCHA triggers — no automated way around it

# When would you use Playwright + BS4 vs just BS4?
# - Just BS4: static HTML pages (HackerNews, Wikipedia)
# - Playwright + BS4: JavaScript-rendered pages (Amazon, Twitter)