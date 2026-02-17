from .db_connection import get_connection

def init_db(db_path="data/instasense.db", reset=True):
    conn = get_connection(db_path)
    if reset:
        conn.execute("DROP TABLE IF EXISTS posts")
    conn.execute("""CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER UNIQUE, username TEXT, category TEXT,
        caption TEXT, likes INTEGER, comments INTEGER, post_date TEXT,
        sentiment_label TEXT, sentiment_score REAL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON posts(category)")
    conn.commit(); conn.close()