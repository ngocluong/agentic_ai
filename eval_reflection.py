import json
from ai_businness_agent import agent

TEST_CASES = [
    {"question": "What is the total revenue for USA?",       "expected":  {'USA': 370000}},
    {"question": "Which category has the highest revenue?",  "expected": {'top_category': "Electronics", "revenue": 430000}},
    {"question": "What is revenue by segment?",              "expected": {'Enterprise': 430000, 'SMB': 135000, 'Consumer': 115000}},
    {"question": "What is total revenue for Canada?",        "expected": {'Canada': 175000}},
    {"question": "What is revenue for Germany?",             "expected": {'Germany': 135000}},
]

passed = 0
failed = 0

# reflected data from ai_businness_agent.py
for test_case in TEST_CASES:
    question = test_case["question"]
    expected = test_case["expected"]
    raw = agent(question)

    try:
        result = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        result = raw

    ok = result == expected
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1

    print(f"[{status}] {question}")
    if not ok:
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")

print(f"\n{passed}/{passed + failed} passed")

passed = 0
failed = 0

# with agent reflected code
for test_case in TEST_CASES:
    question = test_case["question"]
    expected = test_case["expected"]
    raw = agent(question)

    i = 0
    while i < 2:
        try:
            result = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            result = raw

        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
            i = 2
        else:
            failed += 1
            raw = agent(f"Your previous answer was {raw}. Is this complete and correctly formatted JSON? If not, try again.")
            i += 1

    print(f"[{status}] {question}")

print(f"\n{passed}/{passed + failed} passed")
