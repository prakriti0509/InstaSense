"""
InstaSense - Synthetic Instagram Dataset Generator
Generates 7500 realistic Instagram post records with
contextually meaningful captions and realistic engagement patterns.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import csv
import os

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Category configuration with engagement multipliers ───────────────────────
CATEGORIES = {
    "Fashion": {
        "multiplier": 2.2,
        "base_likes": (800, 15000),
        "base_comments": (30, 400),
    },
    "Travel": {
        "multiplier": 2.0,
        "base_likes": (700, 12000),
        "base_comments": (25, 350),
    },
    "Food": {
        "multiplier": 1.6,
        "base_likes": (400, 8000),
        "base_comments": (20, 250),
    },
    "Fitness": {
        "multiplier": 1.5,
        "base_likes": (350, 7000),
        "base_comments": (15, 200),
    },
    "Tech": {
        "multiplier": 1.2,
        "base_likes": (200, 5000),
        "base_comments": (10, 180),
    },
    "Lifestyle": {
        "multiplier": 1.4,
        "base_likes": (300, 6000),
        "base_comments": (15, 220),
    },
}

# ── Realistic usernames ────────────────────────────────────────────────────────
USERNAME_PREFIXES = [
    "the", "its", "im", "hey", "official", "real", "just", "daily",
    "wild", "urban", "modern", "classic", "elite", "pure", "raw",
]
USERNAME_NOUNS = [
    "wanderer", "creator", "dreamer", "explorer", "visionary", "guru",
    "ninja", "maven", "addict", "lover", "nerd", "geek", "pro", "boss",
    "stylist", "foodie", "traveler", "yogi", "coder", "blogger",
]

# ── Caption templates per category ────────────────────────────────────────────
CAPTIONS = {
    "Fashion": {
        "positive": [
            "Obsessed with this new look! 🔥 Confidence is the best outfit you can wear. #OOTD #FashionForward",
            "New collection just dropped and I can't stop wearing it every single day! 💫 #StyleDiaries",
            "This fit is everything. When your outfit matches your mood perfectly. ✨ #FashionInspo",
            "Wore this to brunch and got so many compliments! Style is a form of self-expression. 👗 #Lookbook",
            "Living my best fashionable life! This season's trends are absolutely incredible. 🌟 #FashionWeek",
            "Found this gem at a vintage store and styled it my own way. Thrifting wins again! 💚 #SustainableFashion",
            "Monochrome moment because sometimes less really is more. Clean, minimal, perfect. 🤍 #MinimalistStyle",
            "This designer piece was worth every penny. Investment dressing is the future. 👠 #LuxuryFashion",
            "Street style game on another level today. The city is my runway! 🏙️ #StreetStyle",
            "New season, new wardrobe staples. These pieces will never go out of style. 🎀 #TimelessFashion",
        ],
        "negative": [
            "Fashion week was a bit disappointing this season. Hoping for better next time. 😕 #FashionCritique",
            "Ordered this online and the quality is honestly terrible. Not worth the hype at all. 😤 #Disappointed",
            "Tried to recreate a trending look but it just didn't work for my body type. Frustrating. 😞",
            "Fast fashion waste is getting out of hand. This industry needs a serious reality check. 🌍 #SlowFashion",
        ],
        "neutral": [
            "Outfit of the day featuring some new spring arrivals. Let me know your thoughts below! 👇 #OOTD",
            "Trying out a new style direction this month. Still figuring out what works for me. #FashionJourney",
            "Here's what I wore to the event last weekend. It was a simple but put-together look. 📸 #EventStyle",
        ],
    },
    "Travel": {
        "positive": [
            "Woke up to this view and I genuinely forgot all my worries. Travel heals everything. 🌅 #Wanderlust",
            "This hidden gem in Bali was the most magical place I've ever visited! Can't believe it exists. 🌺 #Bali",
            "Road trip complete! 2000 miles and countless memories. America's beauty is unreal. 🚗 #RoadTrip",
            "Lost in the streets of Tokyo and loving every second. This city has my whole heart. 🗼 #Tokyo",
            "Santorini sunsets hit differently. If you haven't been, put this on your bucket list NOW. 🌇 #Greece",
            "Explored an ancient temple today that's been standing for 2000 years. Perspective is everything. 🏛️ #History",
            "The locals here made us feel so welcome. Travel teaches you that humans are fundamentally kind. 🤝 #Culture",
            "Adventure called, I answered. Paragliding over the Alps was the best decision of my life. 🪂 #Alps",
            "This remote village has no wifi and honestly it's the most connected I've felt in years. 🌿 #DigitalDetox",
            "First time in Morocco and every single sense is overwhelmed in the best possible way. 🕌 #Marrakech",
        ],
        "negative": [
            "This travel company completely ruined our vacation. Zero stars. Demand a refund. 😡 #BadExperience",
            "Overtourism has completely destroyed this once-beautiful destination. So disappointing. 😔 #Overtourism",
            "Missed my connecting flight due to poor planning by the airline. Worst travel day ever. ✈️😤",
            "The hotel was nothing like the photos online. Total scam. Always read reviews carefully. 😠",
        ],
        "neutral": [
            "Day 3 of our European road trip. Sharing the route for anyone planning something similar. 🗺️ #TravelGuide",
            "Packing for a 2-week trip. Carry-on only challenge — wish me luck! ✈️ #PackingTips",
            "Layover in Dubai for 8 hours. Exploring what the airport has to offer. #Transit",
        ],
    },
    "Food": {
        "positive": [
            "This homemade pasta recipe took 3 attempts to perfect and it was SO worth it! 🍝 #HomeCooking",
            "Found the best ramen spot hidden in a tiny alley. The broth is life-changing. 🍜 #FoodieFinds",
            "Sunday brunch perfection achieved. Avocado toast with a poached egg on sourdough. 🥑 #BrunchGoals",
            "Tried making croissants from scratch and I'm genuinely so proud of how they turned out! 🥐 #Baking",
            "This farm-to-table restaurant changed how I think about food. Every ingredient matters. 🌱 #FarmToTable",
            "Chocolate lava cake that took 20 minutes to make. Dessert doesn't have to be complicated! 🍫 #EasyRecipes",
            "Street food in Bangkok is absolutely unbeatable. For $1 you get the best pad thai of your life. 🍛 #Bangkok",
            "Meal prepped for the entire week in just 2 hours. Healthy eating made actually easy! 🥗 #MealPrep",
            "This vegan cheese board is proof that plant-based eating can be absolutely incredible. 🧀🌿 #Vegan",
            "Three-course dinner made entirely from scratch for date night. Cooking is the ultimate love language. ❤️",
        ],
        "negative": [
            "Waited 45 minutes for food that was cold and overpriced. This place is seriously overrated. 😤 #BadFood",
            "This recipe was a disaster. Wasted ingredients and 2 hours of my life. Never again. 😞 #CookingFail",
            "Food delivery took 90 minutes and arrived completely wrong. So frustrated right now. 😡",
            "This trendy restaurant has amazing aesthetics but the food is genuinely terrible. Style over substance. 👎",
        ],
        "neutral": [
            "Trying out a new recipe this weekend. Will share results with you all on Sunday! 🍳 #CookingAdventures",
            "What's everyone's go-to weeknight dinner? Looking for quick ideas that are actually filling. 🤔",
            "Visited a new restaurant downtown. Mixed feelings — the appetizers were great but mains were average. 🍽️",
        ],
    },
    "Fitness": {
        "positive": [
            "New personal record on the squat rack today! 6 months of consistent training paying off. 💪 #GainsTrain",
            "Morning run at sunrise is the most underrated way to start your day. Try it for a week! 🏃 #FitLife",
            "Completed my first triathlon! What seemed impossible 8 months ago is now a reality. 🏅 #Triathlon",
            "Yoga practice is transforming not just my body but my entire mental state. Movement is medicine. 🧘 #Yoga",
            "Hit a weight loss milestone today! Consistency and patience are literally everything. 🎯 #WeightLoss",
            "HIIT workout done before 7am. This discipline is building the life I actually want. ⚡ #HIIT",
            "Rock climbing for the first time and I'm completely hooked! Upper body workout is no joke. 🧗 #Climbing",
            "Swimming 40 laps today. There's something incredibly meditative about being in the water. 🏊 #Swimming",
            "Rest day isn't laziness — it's strategy. Recovery is where the real gains happen. 😴 #RecoveryDay",
            "Trained through depression and anxiety and it genuinely saved me. Movement is mental health. 💚 #MentalHealth",
        ],
        "negative": [
            "Injured my knee again. So frustrated with this setback after all the progress I've made. 😢 #Injury",
            "This fitness influencer's program is completely ineffective. Don't waste your money. 😤 #FitnessScam",
            "Gym was so overcrowded and the equipment was broken. Can't get a decent workout in. 😠",
            "Burned out from overtraining this month. Listening to my body now means resting whether I like it or not. 😔",
        ],
        "neutral": [
            "Starting a new 8-week program today. Will be documenting the full journey here. 📊 #FitnessJourney",
            "Switching from weights to calisthenics this month. Curious to see how my body responds. 🤸 #Calisthenics",
            "New gym gear arrived. Whether the gear actually matters is debatable but I feel ready! 🎽 #GymWear",
        ],
    },
    "Tech": {
        "positive": [
            "Just built my first AI model that actually works in production. The future is genuinely wild. 🤖 #MachineLearning",
            "This new productivity app completely changed how I manage my time. Can't believe I lived without it. ⚡ #Productivity",
            "Open source community came through again. Built something incredible with free tools. 🙌 #OpenSource",
            "Launched my first mobile app today! 8 months of nights and weekends finally shipped. 🚀 #AppLaunch",
            "The new M4 chip performance benchmarks are absolutely mind-blowing. Apple engineering is insane. 💻 #Apple",
            "Finally understand neural networks after this amazing course. The math clicked overnight. 🧠 #DeepLearning",
            "Automated my entire morning routine with some simple Python scripts. 30 minutes saved daily! 🐍 #Automation",
            "WebAssembly is opening up a whole new era of browser applications. This is genuinely exciting. 🌐 #WASM",
            "The new GitHub Copilot update is incredible. AI pair programming is the real deal now. 👨‍💻 #AI",
            "Just deployed my first Kubernetes cluster and somehow everything actually worked first try! ☸️ #DevOps",
        ],
        "negative": [
            "Another major data breach affecting millions of users. Privacy in tech is completely broken. 😡 #DataPrivacy",
            "This framework has the worst documentation I've ever seen. Developers deserve better. 😤 #DevExperience",
            "6 hours debugging this one issue and it was a missing semicolon. I need a career change. 💀 #DebuggingHell",
            "Big tech layoffs are getting really scary. The industry needs to be more humane. 😔 #TechLayoffs",
        ],
        "neutral": [
            "Comparing React vs Vue for a new project. Both have solid arguments in 2025. What do you use? 💭 #WebDev",
            "Setting up a home lab this weekend. Will document the entire build process for anyone interested. 🖥️ #HomeLab",
            "Reading this new paper on transformer architectures. Dense but fascinating material. 📄 #Research",
        ],
    },
    "Lifestyle": {
        "positive": [
            "Journaling every morning for 90 days changed my life in ways I genuinely didn't expect. 📓 #JournalYourLife",
            "Decluttered my entire apartment and the mental clarity that followed was absolutely immediate. ✨ #Minimalism",
            "Planted my first balcony garden and growing your own food hits completely differently. 🌱 #UrbanGarden",
            "Read 3 books this month by swapping just one hour of scrolling per day. Trade-offs matter. 📚 #ReadingChallenge",
            "Morning routine overhaul complete. 5am wake-up, no phone, tea, journal. Life-changing. 🌅 #MorningRoutine",
            "Adopted a rescue dog last month and the joy this animal brings is absolutely indescribable. 🐕 #DogMom",
            "Financial freedom journey update: debt-free after 3 years of intentional living! 💰 #FinancialFreedom",
            "Digital detox weekend was the reset my brain desperately needed. Try it for 48 hours. 📵 #DigitalWellness",
            "Volunteered at a local shelter today. Giving back genuinely fills your soul. ❤️ #GivingBack",
            "Learned to say no this year and it absolutely transformed my relationships and energy. 🙅 #Boundaries",
        ],
        "negative": [
            "Burnout hit me hard this month. Hustle culture is genuinely toxic and I fell for it. 😔 #Burnout",
            "This self-help book is just rehashed common sense wrapped in expensive marketing. Save your money. 😤",
            "Work-life balance is a complete myth when you're building something. Something always suffers. 😞",
            "Social comparison is destroying my mental health. Need to step back from all of this. 😢 #SocialMedia",
        ],
        "neutral": [
            "Trying a new morning routine this week. I'll report back whether it's actually sustainable. ☕ #MorningVibes",
            "Reorganizing my home office setup for better productivity. Small changes, big impact hopefully. 🖥️",
            "Month 2 of my reading challenge — sharing my honest reviews below. Some hits, some misses. 📖 #BookReview",
        ],
    },
}


def generate_username() -> str:
    """Generate a realistic Instagram-style username."""
    pattern = random.choice(["prefix_noun", "noun_numbers", "noun_word", "word_noun"])
    if pattern == "prefix_noun":
        return f"{random.choice(USERNAME_PREFIXES)}_{random.choice(USERNAME_NOUNS)}"
    elif pattern == "noun_numbers":
        return f"{random.choice(USERNAME_NOUNS)}{random.randint(10, 999)}"
    elif pattern == "noun_word":
        words = ["official", "real", "xo", "hq", "studio", "co", "daily"]
        return f"{random.choice(USERNAME_NOUNS)}_{random.choice(words)}"
    else:
        adjectives = ["golden", "silver", "wild", "free", "bold", "calm", "bright", "dark"]
        return f"{random.choice(adjectives)}_{random.choice(USERNAME_NOUNS)}"


def pick_caption(category: str) -> tuple[str, str]:
    """Pick a caption and return (caption, sentiment_hint) pair."""
    weights = [0.60, 0.20, 0.20]  # positive, negative, neutral
    sentiment_type = random.choices(["positive", "negative", "neutral"], weights=weights)[0]
    captions = CAPTIONS[category][sentiment_type]
    return random.choice(captions), sentiment_type


def generate_engagement(category: str, sentiment_hint: str) -> tuple[int, int]:
    """Generate likes and comments based on category and sentiment."""
    cfg = CATEGORIES[category]
    like_min, like_max = cfg["base_likes"]
    comment_min, comment_max = cfg["base_comments"]

    # Sentiment boosts positive engagement
    sentiment_boost = 1.3 if sentiment_hint == "positive" else (0.7 if sentiment_hint == "negative" else 1.0)

    # Add natural variance with a log-normal distribution for realism
    likes = int(np.random.lognormal(
        mean=np.log((like_min + like_max) / 2 * sentiment_boost),
        sigma=0.5
    ))
    likes = max(like_min // 2, min(likes, like_max * 2))

    comments = int(np.random.lognormal(
        mean=np.log((comment_min + comment_max) / 2 * sentiment_boost),
        sigma=0.4
    ))
    comments = max(comment_min // 2, min(comments, comment_max * 2))

    return likes, comments


def generate_post_date(start_days_ago: int = 365) -> str:
    """Generate a random post date within the last N days."""
    days_ago = random.randint(0, start_days_ago)
    post_date = datetime.now() - timedelta(days=days_ago)
    return post_date.strftime("%Y-%m-%d")


def generate_dataset(num_records: int = 7500, output_path: str = "data/instagram_posts.csv") -> pd.DataFrame:
    """
    Generate the full synthetic Instagram dataset.

    Args:
        num_records: Number of rows to generate
        output_path: Path to save the CSV file

    Returns:
        DataFrame with the generated data
    """
    print(f"🔄 Generating {num_records} synthetic Instagram posts...")

    # Category distribution (Fashion & Travel get more posts)
    category_weights = [0.22, 0.22, 0.15, 0.14, 0.13, 0.14]
    categories = list(CATEGORIES.keys())

    records = []
    for i in range(num_records):
        category = random.choices(categories, weights=category_weights)[0]
        caption, sentiment_hint = pick_caption(category)
        likes, comments = generate_engagement(category, sentiment_hint)
        username = generate_username()
        post_date = generate_post_date()

        records.append({
            "post_id": i + 1,
            "username": username,
            "category": category,
            "caption": caption,
            "likes": likes,
            "comments": comments,
            "post_date": post_date,
        })

        if (i + 1) % 1000 == 0:
            print(f"   ✅ Generated {i + 1}/{num_records} records...")

    df = pd.DataFrame(records)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Category distribution:\n{df['category'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    df = generate_dataset(num_records=7500, output_path="data/instagram_posts.csv")
    print("\nSample records:")
    print(df.head(3).to_string())