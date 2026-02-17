import sqlite3
from sentiment.bert_sentiment import analyze_sentiment

def process_sentiment():
    conn = sqlite3.connect("instasense.db")
    cursor = conn.cursor()

    cursor.execute("SELECT post_id, caption FROM posts")
    rows = cursor.fetchall()

    for post_id, caption in rows:
        label, score = analyze_sentiment(caption)

        cursor.execute("""
            UPDATE posts
            SET sentiment_label = ?, sentiment_score = ?
            WHERE post_id = ?
        """, (label, score, post_id))

    conn.commit()
    conn.close()

    print("Sentiment analysis completed successfully!")

if __name__ == "__main__":
    process_sentiment()
