"""
Tests for api/db.py — the SQLite database layer.

Each test gets a fresh isolated database via the temp_db fixture (conftest.py).
"""
from unittest.mock import patch
from api.db import save_message, get_last_messages, save_summary, get_latest_summary


def test_save_and_retrieve_message(temp_db):
    """Saving a message and reading it back returns the same content."""
    with patch("api.db.DB_PATH", temp_db):
        save_message("conv1", "friend", "hello there")
        messages = get_last_messages("conv1", n=5)

    assert len(messages) == 1
    assert messages[0] == ("friend", "hello there")


def test_messages_returned_in_chronological_order(temp_db):
    """get_last_messages returns oldest-first (chronological), not newest-first."""
    with patch("api.db.DB_PATH", temp_db):
        save_message("conv1", "friend", "first")
        save_message("conv1", "user", "second")
        save_message("conv1", "friend", "third")
        messages = get_last_messages("conv1", n=10)

    senders = [m[0] for m in messages]
    contents = [m[1] for m in messages]
    assert contents == ["first", "second", "third"]
    assert senders == ["friend", "user", "friend"]


def test_get_last_messages_respects_limit(temp_db):
    """n parameter caps the number of messages returned."""
    with patch("api.db.DB_PATH", temp_db):
        for i in range(10):
            save_message("conv1", "friend", f"message {i}")
        messages = get_last_messages("conv1", n=3)

    assert len(messages) == 3


def test_conversations_are_isolated(temp_db):
    """Messages from one conversation_id don't appear in another."""
    with patch("api.db.DB_PATH", temp_db):
        save_message("conv-A", "friend", "message in A")
        save_message("conv-B", "friend", "message in B")

        messages_a = get_last_messages("conv-A", n=10)
        messages_b = get_last_messages("conv-B", n=10)

    assert len(messages_a) == 1
    assert messages_a[0][1] == "message in A"
    assert len(messages_b) == 1
    assert messages_b[0][1] == "message in B"


def test_get_last_messages_empty_conversation(temp_db):
    """Querying a conversation that has no messages returns an empty list."""
    with patch("api.db.DB_PATH", temp_db):
        messages = get_last_messages("nonexistent-conv", n=10)

    assert messages == []


def test_save_and_retrieve_summary(temp_db):
    """Saving a summary and fetching it back returns the correct text."""
    with patch("api.db.DB_PATH", temp_db):
        save_summary("conv1", "They were talking about a delayed package.")
        result = get_latest_summary("conv1")

    assert result == "They were talking about a delayed package."


def test_get_latest_summary_returns_empty_string_when_none(temp_db):
    """get_latest_summary returns empty string (not None) for unknown conversations."""
    with patch("api.db.DB_PATH", temp_db):
        result = get_latest_summary("no-summary-here")

    assert result == ""
