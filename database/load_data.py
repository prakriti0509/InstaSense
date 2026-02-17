import pandas as pd
from .db_connection import get_connection

def insert_posts(df, db_path="data/instasense.db"):
    conn = get_connection(db_path)
    inserted = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""INSERT OR IGNORE INTO posts
                (post_id,username,category,caption,likes,comments,post_date,sentiment_label,sentiment_score)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (int(row.get("post_id",0)), str(row.get("username","")),
                 str(row.get("category","")), str(row.get("caption","")),
                 int(row.get("likes",0)), int(row.get("comments",0)),
                 str(row.get("post_date","")), str(row.get("sentiment_label","")),
                 float(row.get("sentiment_score",0.0))))
            inserted += conn.execute("SELECT changes()").fetchone()[0]
        except Exception as e:
            print(f"Skipping row: {e}")
    conn.commit(); conn.close()
    return inserted

def load_posts(db_path="data/instasense.db"):
    conn = get_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM posts", conn)
    conn.close()
    return df