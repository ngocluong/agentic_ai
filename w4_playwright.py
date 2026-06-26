from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://duckduckgo.com/", wait_until="networkidle")
    page.locator('input[name="q"]').fill('AI Agent 2025')
    page.locator('input[name="q"]').press("Enter")
    page.wait_for_selector('[data-testid="result"]', timeout=15000)
    page.screenshot(path='searching_result_screen.png')
    results = page.locator('[data-testid="result"]').all()
    
    for result in results:
        title = result.locator('[data-testid="result-title-a"]').inner_text() if result.locator('[data-testid="result-title-a"]').count() > 0 else "No title"
        link = result.locator('[data-testid="result-extras-url-link"]').inner_text() if result.locator('[data-testid="result-extras-url-link"]').count() > 0 else "No link"
        print(f"Title: {title}\nLink: {link}\n")
    
    page.locator('[data-testid="result-extras-url-link"]').first.click()
    page.wait_for_selector('h1', timeout=15000)
    title = page.locator('h1').first.inner_text() if page.locator('h1').count() > 0 else "No title"
    print(f"Title of the First page: {title}")
    page.screenshot(path='first_page.png')
    page.go_back()
    page.wait_for_load_state("networkidle")
    print(f"Back to: {page.title()}")
    browser.close()
