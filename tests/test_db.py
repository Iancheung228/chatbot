"""
Tests for api/db.py — the SQLite database layer.

Each test gets a fresh isolated database via the temp_db fixture (conftest.py).
"""
import sqlite3
import pytest
from unittest.mock import patch
from api.db import (
    save_message,
    get_last_messages,
    save_summary,
    get_latest_summary,
    log_llm_suggestion,
    mark_suggestion_sent,
    init_db,
)


# ---------------------------------------------------------------------------
# Existing message tests (unchanged behaviour)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# New column: source
# ---------------------------------------------------------------------------

def test_save_message_with_source_stored(temp_db):
    """source kwarg is persisted and can be read back via raw SQL."""
    with patch("api.db.DB_PATH", temp_db):
        save_message("conv1", "user", "hey", source="llm_modified")
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT source FROM messages WHERE conversation_id='conv1'"
            ).fetchone()
    assert row is not None
    assert row[0] == "llm_modified"


def test_save_message_source_defaults_to_null(temp_db):
    """When source is omitted, the column is NULL."""
    with patch("api.db.DB_PATH", temp_db):
        save_message("conv1", "friend", "hi")
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT source FROM messages WHERE conversation_id='conv1'"
            ).fetchone()
    assert row[0] is None


# ---------------------------------------------------------------------------
# LLM suggestion logging
# ---------------------------------------------------------------------------

def test_log_llm_suggestion_inserts_with_sent_zero(temp_db):
    """log_llm_suggestion stores sender='llm' and sent=0."""
    with patch("api.db.DB_PATH", temp_db):
        sid = log_llm_suggestion("conv1", "你好，有什么需要帮忙的吗？")
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT sender, content, sent FROM messages WHERE id=?", (sid,)
            ).fetchone()
    assert row is not None
    assert row[0] == "llm"
    assert row[1] == "你好，有什么需要帮忙的吗？"
    assert row[2] == 0


def test_log_llm_suggestion_returns_unique_ids(temp_db):
    """Each call returns a different id."""
    with patch("api.db.DB_PATH", temp_db):
        id1 = log_llm_suggestion("conv1", "suggestion one")
        id2 = log_llm_suggestion("conv1", "suggestion two")
    assert id1 != id2


# ---------------------------------------------------------------------------
# mark_suggestion_sent
# ---------------------------------------------------------------------------

def test_mark_suggestion_sent_updates_to_sent_one(temp_db):
    """After marking, the row has sent=1."""
    with patch("api.db.DB_PATH", temp_db):
        sid = log_llm_suggestion("conv1", "great reply")
        found = mark_suggestion_sent(sid)
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT sent FROM messages WHERE id=?", (sid,)
            ).fetchone()
    assert found is True
    assert row[0] == 1


def test_mark_suggestion_sent_returns_false_for_nonexistent_id(temp_db):
    """A suggestion_id that doesn't exist returns False without raising."""
    with patch("api.db.DB_PATH", temp_db):
        found = mark_suggestion_sent(99999)
    assert found is False


def test_mark_suggestion_sent_returns_false_if_already_sent(temp_db):
    """Double-accepting the same suggestion_id returns False on the second call."""
    with patch("api.db.DB_PATH", temp_db):
        sid = log_llm_suggestion("conv1", "a reply")
        first  = mark_suggestion_sent(sid)
        second = mark_suggestion_sent(sid)
    assert first  is True
    assert second is False


def test_mark_suggestion_sent_only_affects_target_row(temp_db):
    """Marking one suggestion sent doesn't affect other suggestions."""
    with patch("api.db.DB_PATH", temp_db):
        sid1 = log_llm_suggestion("conv1", "reply one")
        sid2 = log_llm_suggestion("conv1", "reply two")
        mark_suggestion_sent(sid1)
        with sqlite3.connect(temp_db) as conn:
            row2 = conn.execute(
                "SELECT sent FROM messages WHERE id=?", (sid2,)
            ).fetchone()
    assert row2[0] == 0  # sid2 must still be pending


# ---------------------------------------------------------------------------
# get_last_messages excludes sent=0 rows
# ---------------------------------------------------------------------------

def test_get_last_messages_excludes_pending_suggestions(temp_db):
    """Unaccepted LLM suggestions (sent=0) are invisible to get_last_messages."""
    with patch("api.db.DB_PATH", temp_db):
        save_message("conv1", "friend", "how are you?")
        log_llm_suggestion("conv1", "I am fine, thanks!")  # sent=0
        messages = get_last_messages("conv1", n=10)

    assert len(messages) == 1
    assert messages[0] == ("friend", "how are you?")


def test_get_last_messages_includes_accepted_llm_suggestion(temp_db):
    """A suggestion marked sent=1 appears in get_last_messages as sender='llm'."""
    with patch("api.db.DB_PATH", temp_db):
        save_message("conv1", "friend", "how are you?")
        sid = log_llm_suggestion("conv1", "I am fine, thanks!")
        mark_suggestion_sent(sid)
        messages = get_last_messages("conv1", n=10)

    assert len(messages) == 2
    assert messages[1] == ("llm", "I am fine, thanks!")


def test_get_last_messages_excludes_rejected_when_accepted_exists(temp_db):
    """When one suggestion is accepted and one rejected, only the accepted one shows."""
    with patch("api.db.DB_PATH", temp_db):
        save_message("conv1", "friend", "hey")
        log_llm_suggestion("conv1", "rejected reply")   # stays sent=0
        sid = log_llm_suggestion("conv1", "accepted reply")
        mark_suggestion_sent(sid)
        messages = get_last_messages("conv1", n=10)

    contents = [m[1] for m in messages]
    assert "rejected reply" not in contents
    assert "accepted reply" in contents


# ---------------------------------------------------------------------------
# Migration: existing DB without new columns
# ---------------------------------------------------------------------------

def test_migration_adds_sent_and_source_to_legacy_db(tmp_path):
    """
    init_db() must migrate a database that predates the sent/source columns.
    Existing rows must be readable (backfilled sent=1) after migration.
    """
    db_file = str(tmp_path / "legacy.db")

    # Create a legacy-schema DB (no sent or source columns)
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                sender TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES (?, ?, ?)",
            ("c1", "friend", "legacy message"),
        )
        conn.commit()

    with patch("api.db.DB_PATH", db_file):
        init_db()
        messages = get_last_messages("c1", n=5)

    # Legacy row must survive migration with sent=1 (backfilled)
    assert len(messages) == 1
    assert messages[0] == ("friend", "legacy message")


def test_migration_is_idempotent(temp_db):
    """Calling init_db() twice on an already-migrated DB does not raise."""
    with patch("api.db.DB_PATH", temp_db):
        init_db()  # second call — columns already exist, should be silent
