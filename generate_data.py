import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np

categories = {
    "Travel": 8000,
    "Fashion": 10000,
    "Fitness": 6000,
    "Food": 5000,
    "Tech": 4000
}

positive = [
    "Absolutely loved this experience!",
    "Best day ever!",
    "Feeling grateful and blessed.",
    "So happy right now!",
]

neutral = [
    "New post.",
    "Latest update.",
    "Check this out.",
]

negative = [
    "Not my best day.",
    "Could have been better.",
    "A bit disappointed.",
]

def generate_caption(category):
    sentiment_type = random.choice(["positive", "neutral", "negative"])
    
    if sentiment_type == "positive":
        text = random.choice(positive)
    elif sentiment_type == "neutral":
        text = random.choice(neutral)
    else:
        text = random.choice(negative)
        
    return f"{category} vibes ✨ {text}", sentiment_type

data = []

for i in range(1000):
    category = random.choice(list(categories.keys()))
    caption, sentiment = generate_caption(category)
    
    base = categories[category]

    if sentiment == "positive":
        likes = int(np.random.normal(base * 1.1, 500))
    elif sentiment == "negative":
        likes = int(np.random.normal(base * 0.8, 500))
    else:
        likes = int(np.random.normal(base, 500))
        
    comments = int(likes * random.uniform(0.04, 0.08))
    
    date = datetime.now() - timedelta(days=random.randint(0, 365))

    data.append([
        i+1,
        f"user_{random.randint(1,50)}",
        category,
        caption,
        max(likes,100),
        max(comments,10),
        date.date()
    ])

df = pd.DataFrame(data, columns=[
    "post_id",
    "username",
    "category",
    "caption",
    "likes",
    "comments",
    "post_date"
])

df.to_csv("instagram_synthetic_data.csv", index=False)

print("Dataset created successfully!")
