"""
Tests for api/main.py — the FastAPI HTTP layer.

Uses FastAPI's TestClient to make real HTTP requests against the app
without needing a running server. LLM calls are mocked so tests run
offline with no API keys.
"""
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# --- Utility endpoints ---

def test_health_returns_200():
    """GET /health should always return 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_200():
    """GET / should return 200."""
    response = client.get("/")
    assert response.status_code == 200


# --- Input validation ---

def test_suggest_reply_rejects_empty_message():
    """Empty message string should be rejected with 422 before hitting the LLM."""
    response = client.post(
        "/suggest_reply",
        json={"message": "", "conversation_id": "abc-123"},
    )
    assert response.status_code == 422


def test_suggest_reply_rejects_empty_conversation_id():
    """Empty conversation_id should be rejected with 422."""
    response = client.post(
        "/suggest_reply",
        json={"message": "hello", "conversation_id": ""},
    )
    assert response.status_code == 422


def test_suggest_reply_rejects_message_over_max_length():
    """Message exceeding 4000 chars should be rejected with 422."""
    response = client.post(
        "/suggest_reply",
        json={"message": "x" * 4001, "conversation_id": "abc-123"},
    )
    assert response.status_code == 422


# --- LLM call (mocked) ---

def test_suggest_reply_returns_reply_from_llm(temp_db):
    """Valid request with mocked LLM should return a reply in JSON."""
    mock_reply = {"reply": "That sounds really tough, how are you feeling?"}

    with patch("api.db.DB_PATH", temp_db), \
         patch("api.main.settings.llm_provider", "openrouter"), \
         patch("api.main.generate_replies", new=AsyncMock(return_value=mock_reply)):

        response = client.post(
            "/suggest_reply",
            json={"message": "I'm so stressed today", "conversation_id": "abc-123"},
        )

    assert response.status_code == 200
    assert response.json()["reply"] == "That sounds really tough, how are you feeling?"


def test_suggest_reply_returns_502_when_llm_fails(temp_db):
    """If the LLM raises an exception, endpoint should return 502 (not 500 traceback)."""
    with patch("api.db.DB_PATH", temp_db), \
         patch("api.main.settings.llm_provider", "openrouter"), \
         patch("api.main.generate_replies", new=AsyncMock(side_effect=Exception("quota exceeded"))):

        response = client.post(
            "/suggest_reply",
            json={"message": "hey", "conversation_id": "abc-123"},
        )

    assert response.status_code == 502
    assert "quota exceeded" in response.json()["detail"]


# --- History endpoint ---

def test_get_history_returns_empty_for_new_conversation(temp_db):
    """New conversation_id should return an empty messages list."""
    with patch("api.db.DB_PATH", temp_db):
        response = client.get("/get_history", params={"conversation_id": "brand-new"})

    assert response.status_code == 200
    assert response.json() == {"messages": []}


def test_get_history_returns_saved_messages(temp_db):
    """Messages saved to DB should appear in /get_history response."""
    with patch("api.db.DB_PATH", temp_db):
        from api.db import save_message
        save_message("conv-hist", "friend", "hey there")
        save_message("conv-hist", "user", "hey!")

        response = client.get("/get_history", params={"conversation_id": "conv-hist"})

    assert response.status_code == 200
    messages = response.json()["messages"]
    assert len(messages) == 2
    assert messages[0] == {"sender": "friend", "content": "hey there"}
    assert messages[1] == {"sender": "user", "content": "hey!"}
