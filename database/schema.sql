CREATE TABLE IF NOT EXISTS posts (
    post_id INTEGER PRIMARY KEY,
    username TEXT,
    category TEXT,
    caption TEXT,
    likes INTEGER,
    comments INTEGER,
    post_date DATE,
    sentiment_label TEXT,
    sentiment_score REAL
);
