import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_query_success():
    with patch("main.agent") as mock_agent:
        mock_agent.return_value = {"answer": {"USA": 370000}, "time_ms": 12, "tools_used": ["revenue_by_country"]}
        response = client.post("/query", json={"question": "what is LLM", "session_id": "session-query-success"})
    data = response.json()
    assert response.status_code == 200
    assert data["answer"] == {"USA": 370000}
    assert data["time_ms"] == 12
    mock_agent.assert_called_once_with("what is LLM")


def test_missing_question():
    with patch("main.agent") as mock_agent:
        response = client.post("/query", json={"ques": "what is LLM", "session_id": "session-missing-question"})
    assert response.status_code == 400
    mock_agent.assert_not_called()


def test_empty_query():
    with patch("main.agent") as mock_agent:
        response = client.post("/query", json={"question": "", "session_id": "session-empty-query"})
    assert response.status_code == 400
    mock_agent.assert_not_called()


def test_missing_session_id():
    with patch("main.agent") as mock_agent:
        response = client.post("/query", json={"question": "what is LLM"})
    assert response.status_code == 400
    mock_agent.assert_not_called()


def test_long_query():
    long_question = "what is LLM " * 15
    with patch("main.agent") as mock_agent:
        mock_agent.return_value = {"answer": {"top_category": "Electronics", "revenue": 360000}, "time_ms": 40, "tools_used": ["top_category"]}
        response = client.post("/query", json={"question": long_question, "session_id": "session-long-query"})
    data = response.json()
    assert response.status_code == 200
    assert data["answer"] == {"top_category": "Electronics", "revenue": 360000}
    assert data["time_ms"] == 40
    mock_agent.assert_called_once_with(long_question.strip())


def test_missing_body():
    response = client.post("/query")
    assert response.status_code == 400


def test_concurrent_requests():
    answers = {
        "question A": {"answer": {"USA": 370000}, "time_ms": 12, "tools_used": ["revenue_by_country"]},
        "question B": {"answer": {"top_category": "Electronics", "revenue": 360000}, "time_ms": 40, "tools_used": ["top_category"]},
    }

    def fake_agent(question):
        time.sleep(0.05)  # simulate work so both requests are in-flight together
        return answers[question]

    def post(question):
        return client.post("/query", json={"question": question, "session_id": f"session-{question}"})

    with patch("main.agent", side_effect=fake_agent):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(post, "question A")
            future_b = executor.submit(post, "question B")
            response_a = future_a.result()
            response_b = future_b.result()

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert response_a.json()["answer"] == answers["question A"]["answer"]
    assert response_b.json()["answer"] == answers["question B"]["answer"]


def test_rate_limit_exceeded():
    with patch("main.agent") as mock_agent:
        mock_agent.return_value = {"answer": {"USA": 370000}, "time_ms": 12, "tools_used": ["revenue_by_country"]}
        session_id = "session-rate-limit"
        for _ in range(10):
            response = client.post("/query", json={"question": "what is LLM", "session_id": session_id})
            assert response.status_code == 200

        limited_response = client.post("/query", json={"question": "what is LLM", "session_id": session_id})

    assert limited_response.status_code == 429
