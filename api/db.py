import sqlite3
import logging
from api.config import settings

logger = logging.getLogger(__name__)

DB_PATH = settings.db_path


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                sender TEXT,
                content TEXT,
                sent INTEGER DEFAULT 1,
                source TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                summary TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS suggestion_scores (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                suggestion_id   INTEGER NOT NULL,
                conversation_id TEXT NOT NULL,
                rhythm          REAL,
                authenticity    REAL,
                momentum        REAL,
                emotional_match REAL,
                hook_quality    REAL,
                ai_naturalness  REAL,
                overall_score   REAL,
                justifications  TEXT,
                judge_model     TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                raw_response    TEXT
            )
        """)
        try:
            c.execute("ALTER TABLE suggestion_scores ADD COLUMN ai_naturalness REAL")
            logger.info("Migration: added column 'ai_naturalness' to suggestion_scores")
        except sqlite3.OperationalError:
            pass  # column already exists
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conversation_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_summaries_conv ON summaries (conversation_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_scores_conv ON suggestion_scores (conversation_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_scores_suggestion ON suggestion_scores (suggestion_id)")

        # Migrations for existing databases that pre-date these columns.
        # ALTER TABLE ADD COLUMN does not set DEFAULT on existing rows in all
        # SQLite versions, so we explicitly backfill NULLs after each migration.
        _migrations = [
            ("sent",   "INTEGER DEFAULT 1", "UPDATE messages SET sent = 1 WHERE sent IS NULL"),
            ("source", "TEXT",              None),
        ]
        for col, definition, backfill_sql in _migrations:
            try:
                c.execute(f"ALTER TABLE messages ADD COLUMN {col} {definition}")
                logger.info("Migration: added column '%s' to messages", col)
                if backfill_sql:
                    c.execute(backfill_sql)
                    logger.info("Migration: backfilled '%s'", col)
            except sqlite3.OperationalError:
                pass  # column already exists — nothing to do

        conn.commit()
        logger.info("Database tables initialized")


def save_message(conversation_id: str, sender: str, content: str,
                 sent: int = 1, source: str | None = None) -> int:
    """Insert a message row. Returns the new row id."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages (conversation_id, sender, content, sent, source)"
            " VALUES (?, ?, ?, ?, ?)",
            (conversation_id, sender, content, sent, source),
        )
        conn.commit()
        row_id = c.lastrowid
        logger.debug(
            "Message saved: id=%s conversation_id=%s sender=%s sent=%s source=%s",
            row_id, conversation_id, sender, sent, source,
        )
    return row_id


def log_llm_suggestion(conversation_id: str, content: str) -> int:
    """
    Persist an LLM-generated suggestion before the user has accepted it.
    Stored as sender='llm', sent=0.
    Returns the new row id (used as suggestion_id in the frontend).
    """
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages (conversation_id, sender, content, sent)"
            " VALUES (?, 'llm', ?, 0)",
            (conversation_id, content),
        )
        conn.commit()
        suggestion_id = c.lastrowid
        logger.debug(
            "LLM suggestion logged: id=%s conversation_id=%s", suggestion_id, conversation_id
        )
    return suggestion_id


def mark_suggestion_sent(suggestion_id: int) -> bool:
    """
    Mark a pending LLM suggestion as accepted (sent=1).
    Only updates rows where sender='llm' AND sent=0 — prevents double-accept.
    Returns True if the row was found and updated, False otherwise.
    """
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE messages SET sent = 1 WHERE id = ? AND sender = 'llm' AND sent = 0",
            (suggestion_id,),
        )
        conn.commit()
        found = c.rowcount > 0
        if not found:
            logger.warning(
                "mark_suggestion_sent: id=%s not found or already sent", suggestion_id
            )
    return found


def get_all_messages(conversation_id: str) -> list[tuple[str, str]]:
    """Return all confirmed (sent=1) messages in chronological order. No limit."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT sender, content FROM messages"
            " WHERE conversation_id = ? AND sent = 1"
            " ORDER BY timestamp ASC, id ASC",
            (conversation_id,),
        )
        return c.fetchall()


def get_last_messages(conversation_id: str, n: int = 10) -> list[tuple[str, str]]:
    """
    Return the last n *sent* messages (sent=1) in chronological order.
    Excludes unaccepted LLM suggestions (sent=0).
    """
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT sender, content FROM messages"
            " WHERE conversation_id = ? AND sent = 1"
            " ORDER BY timestamp DESC, id DESC LIMIT ?",
            (conversation_id, n),
        )
        messages = c.fetchall()
    return messages[::-1]  # reverse to chronological order


def save_summary(conversation_id: str, summary: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO summaries (conversation_id, summary) VALUES (?, ?)",
            (conversation_id, summary),
        )
        conn.commit()


def get_latest_summary(conversation_id: str) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT summary FROM summaries WHERE conversation_id = ?"
            " ORDER BY timestamp DESC LIMIT 1",
            (conversation_id,),
        )
        row = c.fetchone()
    return row[0] if row else ""


def save_suggestion_score(
    suggestion_id: int,
    conversation_id: str,
    scores: dict,
    judge_model: str,
    raw_response: str,
) -> None:
    """Persist LLM judge scores for a suggestion. scores keys: rhythm, authenticity,
    momentum, emotional_match, hook_quality, ai_naturalness, overall_score,
    justifications (JSON str)."""
    import json
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO suggestion_scores
                (suggestion_id, conversation_id, rhythm, authenticity, momentum,
                 emotional_match, hook_quality, ai_naturalness, overall_score,
                 justifications, judge_model, raw_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                suggestion_id,
                conversation_id,
                scores.get("rhythm"),
                scores.get("authenticity"),
                scores.get("momentum"),
                scores.get("emotional_match"),
                scores.get("hook_quality"),
                scores.get("ai_naturalness"),
                scores.get("overall_score"),
                json.dumps(scores.get("justifications", {}), ensure_ascii=False),
                judge_model,
                raw_response,
            ),
        )
        conn.commit()
    logger.debug("Judge score saved: suggestion_id=%s overall=%.1f", suggestion_id, scores.get("overall_score", 0))
