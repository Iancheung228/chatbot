import sqlite3
import logging
from api.config import settings

logger = logging.getLogger(__name__)

DB_PATH = settings.db_path

def init_db():
    # Use 'with' to connect to the SQLite database and automatically close the connection when done
    with sqlite3.connect(DB_PATH) as conn:
        # Create a cursor object 
        c = conn.cursor()

        create_messages_table = """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                sender TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        create_summaries_table = """
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                summary TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        c.execute(create_messages_table)
        c.execute(create_summaries_table)
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conversation_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_summaries_conv ON summaries (conversation_id)")
        conn.commit()
        logger.info("Database tables initialized")

def save_message(conversation_id, sender, content):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        #instead of writing SQL directly in our Python script with hardcoded values, we’ll use parameterized queries to make our code more secure and flexible.
        
        insert_to_msg_table = '''
        INSERT INTO messages (conversation_id, sender, content) VALUES (?, ?, ?)'''

        msg_data = (conversation_id, sender, content)
        c.execute(insert_to_msg_table, msg_data)
        conn.commit()
        logger.debug("Message saved: conversation_id=%s sender=%s", conversation_id, sender)

def get_last_messages(conversation_id, n=10):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT sender, content FROM messages WHERE conversation_id=? ORDER BY timestamp DESC, id DESC LIMIT ?", (conversation_id, n))
        messages = c.fetchall()
    return messages[::-1]  # reverse to chronological order

def save_summary(conversation_id, summary):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO summaries (conversation_id, summary) VALUES (?, ?)", (conversation_id, summary))
        conn.commit()

def get_latest_summary(conversation_id):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT summary FROM summaries WHERE conversation_id=? ORDER BY timestamp DESC LIMIT 1", (conversation_id,))
        row = c.fetchone()
    return row[0] if row else ""