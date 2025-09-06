import sqlite3
from datetime import datetime

DB_PATH = "chatbot.db"

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
        conn.commit()
        print("Tables created successfully!")

def save_message(conversation_id, sender, content):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        #instead of writing SQL directly in our Python script with hardcoded values, we’ll use parameterized queries to make our code more secure and flexible.
        
        insert_to_msg_table = '''
        INSERT INTO messages (conversation_id, sender, content) VALUES (?, ?, ?)'''

        msg_data = (conversation_id, sender, content)
        c.execute(insert_to_msg_table, msg_data)
        conn.commit()
        # No need to call connection.close(); it's done automatically!
        print("Record inserted successfully!")

def get_last_messages(conversation_id, n=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT sender, content FROM messages WHERE conversation_id=? ORDER BY timestamp DESC LIMIT ?", (conversation_id, n))
    messages = c.fetchall()
    conn.close()
    return messages[::-1]  # reverse to chronological order

def save_summary(conversation_id, summary):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO summaries (conversation_id, summary) VALUES (?, ?)", (conversation_id, summary))
    conn.commit()
    conn.close()

def get_latest_summary(conversation_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT summary FROM summaries WHERE conversation_id=? ORDER BY timestamp DESC LIMIT 1", (conversation_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""