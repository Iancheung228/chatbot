"""
Tests for api/main.py — the FastAPI HTTP layer.

Uses FastAPI's TestClient to make real HTTP requests against the app
without needing a running server. LLM calls are mocked so tests run
offline with no API keys.
"""
import sqlite3
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

def test_health_returns_200():
    """GET /health should always return 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_200():
    """GET / should return 200."""
    response = client.get("/")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /friend_message
# ---------------------------------------------------------------------------

def test_friend_message_saves_to_db(temp_db):
    """POST /friend_message saves the message as sender='friend', sent=1."""
    with patch("api.db.DB_PATH", temp_db):
        response = client.post(
            "/friend_message",
            json={"conversation_id": "conv-friend", "content": "hey what's up"},
        )
        assert response.status_code == 200

        from api.db import get_last_messages
        messages = get_last_messages("conv-friend", n=5)

    assert len(messages) == 1
    assert messages[0] == ("friend", "hey what's up")


def test_friend_message_rejects_empty_content(temp_db):
    """Empty content should be rejected with 422."""
    with patch("api.db.DB_PATH", temp_db):
        response = client.post(
            "/friend_message",
            json={"conversation_id": "conv-friend", "content": ""},
        )
    assert response.status_code == 422


def test_friend_message_rejects_empty_conversation_id(temp_db):
    """Empty conversation_id should be rejected with 422."""
    with patch("api.db.DB_PATH", temp_db):
        response = client.post(
            "/friend_message",
            json={"conversation_id": "", "content": "hello"},
        )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Input validation — /suggest_reply
# ---------------------------------------------------------------------------

def test_suggest_reply_rejects_empty_conversation_id():
    """Empty conversation_id should be rejected with 422."""
    response = client.post(
        "/suggest_reply",
        json={"conversation_id": ""},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /suggest_reply — LLM call (mocked)
# ---------------------------------------------------------------------------

def test_suggest_reply_returns_reply_and_suggestion_id(temp_db):
    """Valid request with mocked LLM should return reply and suggestion_id."""
    mock_reply = {"reply": "That sounds really tough, how are you feeling?"}

    with patch("api.db.DB_PATH", temp_db), \
         patch("api.main.settings.llm_provider", "openrouter"), \
         patch("api.main.generate_replies", new=AsyncMock(return_value=mock_reply)):

        response = client.post(
            "/suggest_reply",
            json={"conversation_id": "abc-123"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "That sounds really tough, how are you feeling?"
    assert isinstance(data["suggestion_id"], int)


def test_suggest_reply_saves_suggestion_as_pending(temp_db):
    """
    /suggest_reply must save the LLM suggestion with sent=0 (pending).
    It should NOT appear in get_last_messages (which only returns sent=1).
    """
    mock_reply = {"reply": "Sure, let's catch up!"}

    with patch("api.db.DB_PATH", temp_db), \
         patch("api.main.settings.llm_provider", "openrouter"), \
         patch("api.main.generate_replies", new=AsyncMock(return_value=mock_reply)):

        client.post(
            "/suggest_reply",
            json={"conversation_id": "no-save-conv"},
        )

    with patch("api.db.DB_PATH", temp_db):
        from api.db import get_last_messages
        messages = get_last_messages("no-save-conv", n=10)

        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT sender, sent FROM messages WHERE conversation_id='no-save-conv'"
            ).fetchone()

    # Not visible in confirmed history
    assert messages == []
    # But exists in DB as pending training data
    assert row[0] == "llm"
    assert row[1] == 0


def test_suggest_reply_returns_502_when_llm_fails(temp_db):
    """If the LLM raises an exception, endpoint should return 502."""
    with patch("api.db.DB_PATH", temp_db), \
         patch("api.main.settings.llm_provider", "openrouter"), \
         patch("api.main.generate_replies", new=AsyncMock(side_effect=Exception("quota exceeded"))):

        response = client.post(
            "/suggest_reply",
            json={"conversation_id": "abc-123"},
        )

    assert response.status_code == 502
    assert "quota exceeded" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /send_user_message — source='manual'
# ---------------------------------------------------------------------------

def test_send_user_message_manual_inserts_user_row(temp_db):
    """source='manual' inserts a sender='user' row with sent=1."""
    with patch("api.db.DB_PATH", temp_db):
        response = client.post(
            "/send_user_message",
            json={"conversation_id": "conv1", "content": "I typed this myself", "source": "manual"},
        )
        assert response.status_code == 200

        from api.db import get_last_messages
        messages = get_last_messages("conv1", n=5)

    assert len(messages) == 1
    assert messages[0] == ("user", "I typed this myself")


def test_send_user_message_manual_source_column(temp_db):
    """source='manual' stores 'manual' in the source column."""
    with patch("api.db.DB_PATH", temp_db):
        client.post(
            "/send_user_message",
            json={"conversation_id": "conv1", "content": "typed", "source": "manual"},
        )
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT source FROM messages WHERE conversation_id='conv1'"
            ).fetchone()
    assert row[0] == "manual"


# ---------------------------------------------------------------------------
# /send_user_message — source='llm_accepted'
# ---------------------------------------------------------------------------

def test_send_user_message_llm_accepted_marks_suggestion_sent(temp_db):
    """source='llm_accepted' flips the suggestion row to sent=1."""
    with patch("api.db.DB_PATH", temp_db):
        from api.db import log_llm_suggestion
        sid = log_llm_suggestion("conv2", "Sounds good!")

        send_resp = client.post(
            "/send_user_message",
            json={
                "conversation_id": "conv2",
                "content": "Sounds good!",
                "source": "llm_accepted",
                "suggestion_id": sid,
            },
        )
        assert send_resp.status_code == 200

        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT sender, sent FROM messages WHERE id=?", (sid,)
            ).fetchone()

    assert row[0] == "llm"
    assert row[1] == 1  # now accepted


def test_send_user_message_llm_accepted_appears_in_history(temp_db):
    """After accepting, the suggestion appears in /get_history."""
    with patch("api.db.DB_PATH", temp_db):
        from api.db import log_llm_suggestion
        sid = log_llm_suggestion("conv2", "See you then!")

        client.post(
            "/send_user_message",
            json={
                "conversation_id": "conv2",
                "content": "See you then!",
                "source": "llm_accepted",
                "suggestion_id": sid,
            },
        )
        history = client.get("/get_history", params={"conversation_id": "conv2"})

    messages = history.json()["messages"]
    assert len(messages) == 1
    assert messages[0]["sender"] == "llm"
    assert messages[0]["content"] == "See you then!"


def test_send_user_message_llm_accepted_requires_suggestion_id(temp_db):
    """Missing suggestion_id with source='llm_accepted' returns 422."""
    with patch("api.db.DB_PATH", temp_db):
        response = client.post(
            "/send_user_message",
            json={"conversation_id": "conv3", "content": "ok", "source": "llm_accepted"},
        )
    assert response.status_code == 422


def test_send_user_message_llm_accepted_404_for_wrong_id(temp_db):
    """A suggestion_id that doesn't exist returns 404."""
    with patch("api.db.DB_PATH", temp_db):
        response = client.post(
            "/send_user_message",
            json={
                "conversation_id": "conv3",
                "content": "ok",
                "source": "llm_accepted",
                "suggestion_id": 99999,
            },
        )
    assert response.status_code == 404


def test_send_user_message_llm_accepted_404_on_double_accept(temp_db):
    """Accepting the same suggestion twice returns 404 on the second attempt."""
    with patch("api.db.DB_PATH", temp_db):
        from api.db import log_llm_suggestion
        sid = log_llm_suggestion("conv2", "once")
        payload = {"conversation_id": "conv2", "content": "once", "source": "llm_accepted", "suggestion_id": sid}
        first  = client.post("/send_user_message", json=payload)
        second = client.post("/send_user_message", json=payload)

    assert first.status_code == 200
    assert second.status_code == 404


# ---------------------------------------------------------------------------
# /send_user_message — source='llm_modified'
# ---------------------------------------------------------------------------

def test_send_user_message_llm_modified_inserts_user_row(temp_db):
    """source='llm_modified' inserts a new sender='user' row."""
    with patch("api.db.DB_PATH", temp_db):
        from api.db import log_llm_suggestion
        sid = log_llm_suggestion("conv4", "original text")

        client.post(
            "/send_user_message",
            json={
                "conversation_id": "conv4",
                "content": "modified text",
                "source": "llm_modified",
            },
        )

        from api.db import get_last_messages
        messages = get_last_messages("conv4", n=10)

        with sqlite3.connect(temp_db) as conn:
            original = conn.execute(
                "SELECT sent FROM messages WHERE id=?", (sid,)
            ).fetchone()

    # The user's edited version appears in history
    assert len(messages) == 1
    assert messages[0] == ("user", "modified text")
    # The original LLM suggestion stays sent=0 (preserved for training data)
    assert original[0] == 0


def test_send_user_message_llm_modified_source_column(temp_db):
    """source='llm_modified' stores 'llm_modified' in the source column."""
    with patch("api.db.DB_PATH", temp_db):
        client.post(
            "/send_user_message",
            json={"conversation_id": "conv4", "content": "tweaked", "source": "llm_modified"},
        )
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT source FROM messages WHERE conversation_id='conv4'"
            ).fetchone()
    assert row[0] == "llm_modified"


# ---------------------------------------------------------------------------
# /send_user_message — invalid source
# ---------------------------------------------------------------------------

def test_send_user_message_invalid_source_returns_422(temp_db):
    """An unrecognised source value returns 422."""
    with patch("api.db.DB_PATH", temp_db):
        response = client.post(
            "/send_user_message",
            json={"conversation_id": "conv5", "content": "hi", "source": "alien"},
        )
    assert response.status_code == 422


def test_send_user_message_empty_content_returns_422(temp_db):
    """Empty content string is rejected with 422."""
    with patch("api.db.DB_PATH", temp_db):
        response = client.post(
            "/send_user_message",
            json={"conversation_id": "conv5", "content": "", "source": "manual"},
        )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /get_history
# ---------------------------------------------------------------------------

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


def test_get_history_excludes_unaccepted_suggestions(temp_db):
    """
    LLM suggestions that were never accepted (sent=0) must not appear
    in /get_history — they should be invisible to the chat UI.
    """
    with patch("api.db.DB_PATH", temp_db):
        from api.db import save_message, log_llm_suggestion
        save_message("conv6", "friend", "what do you think?")
        log_llm_suggestion("conv6", "this suggestion was rejected")  # sent=0

        response = client.get("/get_history", params={"conversation_id": "conv6"})

    messages = response.json()["messages"]
    assert len(messages) == 1
    assert messages[0]["content"] == "what do you think?"
