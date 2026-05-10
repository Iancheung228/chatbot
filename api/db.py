import json
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
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                sender          TEXT NOT NULL,
                content         TEXT NOT NULL,
                sent            INTEGER DEFAULT 1,
                source          TEXT,
                pipeline        TEXT,
                reply_model     TEXT,
                step1_json      TEXT,
                step2_json      TEXT,
                timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                summary         TEXT NOT NULL,
                timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # One row per step1 candidate — 3 rows per three_step suggestion
        # selected=1 (chosen) vs selected=0 (rejected) = natural DPO pairs
        c.execute("""
            CREATE TABLE IF NOT EXISTS candidate_directions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id      INTEGER NOT NULL,
                direction_index INTEGER NOT NULL,
                angle           TEXT,
                draft           TEXT,
                selected        INTEGER DEFAULT 0,
                scores_json     TEXT
            )
        """)

        # One row per suggestion — scored every generation regardless of acceptance
        c.execute("""
            CREATE TABLE IF NOT EXISTS judge_scores (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id      INTEGER NOT NULL,
                conversation_id TEXT NOT NULL,
                overall_score   REAL,
                verdict         TEXT,
                scores_json     TEXT,
                justifications  TEXT,
                rubric_version  TEXT,
                judge_model     TEXT,
                raw_response    TEXT,
                pipeline        TEXT,
                reply_model     TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv      ON messages (conversation_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_summaries_conv     ON summaries (conversation_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_candidates_msg     ON candidate_directions (message_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_judge_msg          ON judge_scores (message_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_judge_conv         ON judge_scores (conversation_id)")

        conn.commit()
        logger.info("Database tables initialized")


# ── Messages ──────────────────────────────────────────────────────────────────

def save_message(
    conversation_id: str,
    sender: str,
    content: str,
    sent: int = 1,
    source: str | None = None,
) -> int:
    """Insert a human message row. Returns the new row id."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages (conversation_id, sender, content, sent, source)"
            " VALUES (?, ?, ?, ?, ?)",
            (conversation_id, sender, content, sent, source),
        )
        conn.commit()
        row_id = c.lastrowid
    logger.debug("Message saved: id=%s sender=%s", row_id, sender)
    return row_id


def log_llm_suggestion(
    conversation_id: str,
    content: str,
    pipeline: str = "single",
    reply_model: str | None = None,
    step1: dict | None = None,
    step2: dict | None = None,
) -> int:
    """
    Persist a pending LLM suggestion (sent=0) with full pipeline metadata.
    step1/step2 are stored as JSON; None for single/two_step pipelines.
    Returns the new row id (used as message_id everywhere else).
    """
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages"
            " (conversation_id, sender, content, sent, pipeline, reply_model, step1_json, step2_json)"
            " VALUES (?, 'llm', ?, 0, ?, ?, ?, ?)",
            (
                conversation_id,
                content,
                pipeline,
                reply_model,
                json.dumps(step1, ensure_ascii=False) if step1 else None,
                json.dumps(step2, ensure_ascii=False) if step2 else None,
            ),
        )
        conn.commit()
        message_id = c.lastrowid
    logger.debug("LLM suggestion logged: id=%s pipeline=%s", message_id, pipeline)
    return message_id


def mark_suggestion_sent(suggestion_id: int) -> bool:
    """Mark a pending LLM suggestion as accepted (sent=1)."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE messages SET sent = 1 WHERE id = ? AND sender = 'llm' AND sent = 0",
            (suggestion_id,),
        )
        conn.commit()
        found = c.rowcount > 0
    if not found:
        logger.warning("mark_suggestion_sent: id=%s not found or already sent", suggestion_id)
    return found


def get_all_messages(conversation_id: str) -> list[tuple[str, str]]:
    """Return all confirmed (sent=1) messages in chronological order."""
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
    """Return the last n confirmed (sent=1) messages in chronological order."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT sender, content FROM messages"
            " WHERE conversation_id = ? AND sent = 1"
            " ORDER BY timestamp DESC, id DESC LIMIT ?",
            (conversation_id, n),
        )
        messages = c.fetchall()
    return messages[::-1]


# ── Summaries ─────────────────────────────────────────────────────────────────

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


# ── Candidate directions ──────────────────────────────────────────────────────

def log_candidate_directions(
    message_id: int,
    directions: list[dict],
    evaluations: list[dict],
    selected_index: int | None,
) -> None:
    """
    Insert one row per step1 candidate direction (3 rows per three_step suggestion).
    Merges step1 directions with step2 per-candidate evaluation scores by index.
    selected=1 marks the direction step2 chose; others are selected=0.
    """
    scores_by_index = {e["index"]: e.get("scores", {}) for e in evaluations}
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        for i, d in enumerate(directions):
            c.execute(
                "INSERT INTO candidate_directions"
                " (message_id, direction_index, angle, draft, selected, scores_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    i,
                    d.get("angle"),
                    d.get("draft"),
                    1 if i == selected_index else 0,
                    json.dumps(scores_by_index.get(i, {}), ensure_ascii=False),
                ),
            )
        conn.commit()
    logger.debug("Candidate directions logged: message_id=%s count=%d", message_id, len(directions))


# ── Judge scores ──────────────────────────────────────────────────────────────

def save_judge_score(
    message_id: int,
    conversation_id: str,
    overall_score: float,
    verdict: str | None,
    scores_json: str,
    justifications: str,
    rubric_version: str,
    judge_model: str,
    raw_response: str,
    pipeline: str,
    reply_model: str,
) -> None:
    """
    Persist judge evaluation for one suggestion. Runs for every generation.
    overall_score is always 0-100% regardless of rubric dimension count,
    as long as weights sum to 1.0 and scores are on a 1-3 scale.
    scores_json stores the full dimension breakdown; changing the rubric
    only requires bumping rubric_version — no migration needed.
    """
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO judge_scores
                (message_id, conversation_id,
                 overall_score, verdict, scores_json, justifications,
                 rubric_version, judge_model, raw_response, pipeline, reply_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id, conversation_id,
                overall_score, verdict, scores_json, justifications,
                rubric_version, judge_model, raw_response, pipeline, reply_model,
            ),
        )
        conn.commit()
    logger.debug("Judge score saved: message_id=%s overall=%.1f", message_id, overall_score)
