"""
InstaSense - Demo Account Seeder
Run ONCE with: python seed_demo.py
This creates all 10 demo profiles with posts in the database.
"""

import os, sys, sqlite3, hashlib, datetime, random, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "instasense_users.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def init_db():
    conn = get_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        instagram_handle TEXT,
        is_demo INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS user_posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        caption TEXT, category TEXT,
        likes INTEGER DEFAULT 0, comments INTEGER DEFAULT 0,
        post_date TEXT, sentiment_label TEXT,
        sentiment_score REAL, engagement_score REAL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id))""")
    conn.commit()
    conn.close()
    print("✅ Database schema ready")

# ── Sentiment ─────────────────────────────────────────────────────────────────
EMOJI_SENTIMENT_POS = {"😍":0.9,"🔥":0.8,"💯":0.8,"✨":0.7,"🌟":0.8,"💫":0.7,"🎉":0.8,
    "🙌":0.7,"💪":0.7,"🚀":0.8,"🏆":0.8,"👑":0.8,"💎":0.7,"🌈":0.7,"❤️":0.8,
    "😊":0.7,"😁":0.7,"🤩":0.9,"🥰":0.9,"😘":0.8,"👍":0.7,"🥳":0.8,"💃":0.7}
EMOJI_SENTIMENT_NEG = {"😤":0.8,"😡":0.9,"😠":0.8,"🤬":0.9,"😞":0.8,"😔":0.7,
    "😢":0.8,"😭":0.8,"💀":0.7,"👎":0.8,"😒":0.7,"🙄":0.6,"😩":0.7,"😫":0.7}

def analyze_sentiment(text):
    if not text or not text.strip():
        return "Neutral", 0.5
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        emoji_boost = 0.0
        for e, w in EMOJI_SENTIMENT_POS.items():
            if e in text: emoji_boost += w * text.count(e) * 0.12
        for e, w in EMOJI_SENTIMENT_NEG.items():
            if e in text: emoji_boost -= w * text.count(e) * 0.12
        emoji_boost = max(-0.6, min(0.6, emoji_boost))

        clean = re.sub(r"http\S+|@\w+", "", text)
        clean = re.sub(r"#(\w+)", r"\1", clean)
        clean = clean.encode("ascii", errors="ignore").decode("ascii").strip()

        POS = {"great","obsessed","amazing","love","best","perfect","incredible","wonderful",
               "fantastic","awesome","beautiful","proud","milestone","achieved","joy","grateful",
               "delicious","stunning","worth","magical","excited","happy","blessed","thrilled"}
        NEG = {"terrible","worst","hate","disappointed","frustrating","broken","overpriced",
               "disaster","wasted","scam","ruined","failed","toxic","burnout","injured",
               "setback","overrated","miserable","awful","horrible","annoying","useless"}

        analyzer = SentimentIntensityAnalyzer()
        scores   = analyzer.polarity_scores(clean)
        compound = scores["compound"]
        words    = set(clean.lower().split())
        compound = max(-1.0, min(1.0, compound + len(words&POS)*0.08 - len(words&NEG)*0.08 + emoji_boost))
        if compound >= 0.03:
            return "Positive", round(min(0.5 + compound/2, 0.99), 4)
        else:
            return "Negative", round(min(0.5 + abs(compound)/2, 0.99), 4)
    except ImportError:
        # Simple fallback if vaderSentiment not installed
        pos_words = ["amazing","love","best","perfect","incredible","awesome","beautiful",
                     "proud","obsessed","fantastic","wonderful","grateful","magical","excited"]
        neg_words = ["terrible","worst","hate","disappointed","broken","disaster","wasted",
                     "scam","ruined","failed","toxic","burnout","frustrated","miserable"]
        text_lower = text.lower()
        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)
        if pos_count > neg_count:
            return "Positive", round(0.6 + pos_count * 0.05, 4)
        elif neg_count > pos_count:
            return "Negative", round(0.6 + neg_count * 0.05, 4)
        return "Neutral", 0.5

# ── Demo profiles ─────────────────────────────────────────────────────────────
DEMO_PROFILES = [
    {"username":"fashionista_aria",  "password":"demo1234", "email":"aria@instasense.demo",      "handle":"@aria_styles",    "specialty":"Fashion"},
    {"username":"wanderlust_kai",    "password":"demo1234", "email":"kai@instasense.demo",       "handle":"@kai_travels",    "specialty":"Travel"},
    {"username":"foodie_priya",      "password":"demo1234", "email":"priya@instasense.demo",     "handle":"@priya_eats",     "specialty":"Food"},
    {"username":"fitlife_marcus",    "password":"demo1234", "email":"marcus@instasense.demo",    "handle":"@marcus_fits",    "specialty":"Fitness"},
    {"username":"techbro_sam",       "password":"demo1234", "email":"sam@instasense.demo",       "handle":"@sam_codes",      "specialty":"Tech"},
    {"username":"lifestyle_zoe",     "password":"demo1234", "email":"zoe@instasense.demo",       "handle":"@zoe_vibes",      "specialty":"Lifestyle"},
    {"username":"glamour_nina",      "password":"demo1234", "email":"nina@instasense.demo",      "handle":"@nina_glam",      "specialty":"Fashion"},
    {"username":"explorer_raj",      "password":"demo1234", "email":"raj@instasense.demo",       "handle":"@raj_explores",   "specialty":"Travel"},
    {"username":"chef_isabella",     "password":"demo1234", "email":"isabella@instasense.demo",  "handle":"@bella_cooks",    "specialty":"Food"},
    {"username":"wellness_leo",      "password":"demo1234", "email":"leo@instasense.demo",       "handle":"@leo_wellness",   "specialty":"Fitness"},
]

DEMO_POSTS = {
    "Fashion": [
        ("Obsessed with this new look! 🔥 Confidence is the best outfit. #OOTD #FashionForward", 8400, 310),
        ("New collection just dropped and I can't stop wearing it 💫 #StyleDiaries", 6200, 245),
        ("This fit is everything ✨ When your outfit matches your mood perfectly #FashionInspo", 9100, 380),
        ("Wore this to brunch and got so many compliments! 👗 #Lookbook", 5500, 210),
        ("Street style game on another level today 🏙️ The city is my runway! #StreetStyle", 7300, 290),
        ("Monochrome moment 🤍 Sometimes less really is more. #MinimalistStyle", 4800, 175),
        ("Found this gem at a vintage store 💚 Thrifting wins again! #SustainableFashion", 5100, 198),
        ("This designer piece was worth every penny 👠 Investment dressing. #LuxuryFashion", 11200, 450),
        ("Fashion week disappointment this season 😤 Hoping for better next time. #FashionCritique", 3200, 280),
        ("Ordered this online and the quality is terrible 😞 Not worth the hype. #Disappointed", 2800, 320),
        ("New season wardrobe staples 🎀 These pieces will never go out of style. #TimelessFashion", 6700, 260),
        ("Capsule wardrobe update complete! Less is truly more. 🌿 #CapsuleWardrobe", 5900, 230),
        ("Fast fashion waste is out of hand 🌍 This industry needs a reality check. #SlowFashion", 4100, 410),
        ("Styling tips for petite frames — your body is perfect as it is! ❤️ #BodyPositive", 8800, 520),
        ("Date night outfit locked in 💜 Feeling confident and beautiful tonight! #DateNight", 7600, 340),
    ],
    "Travel": [
        ("Woke up to this view and forgot all my worries 🌅 Travel heals everything. #Wanderlust", 9200, 340),
        ("Hidden gem in Bali was the most magical place I've ever visited! 🌺 #Bali", 11500, 480),
        ("Road trip complete! 2000 miles and countless memories 🚗 #RoadTrip", 7800, 295),
        ("Lost in the streets of Tokyo 🗼 This city has my whole heart. #Tokyo", 10200, 410),
        ("Santorini sunsets hit differently 🌇 Put this on your bucket list NOW. #Greece", 13400, 560),
        ("Explored an ancient temple today — 2000 years old 🏛️ Perspective is everything. #History", 6100, 245),
        ("The locals here made us feel so welcome 🤝 Travel teaches kindness. #Culture", 5400, 310),
        ("This travel company completely ruined our vacation 😡 Demand a refund. #BadExperience", 3800, 490),
        ("Overtourism has destroyed this once-beautiful destination 😔 So disappointing. #Overtourism", 4200, 380),
        ("Paragliding over the Alps was the best decision of my life 🪂 #Alps", 12100, 520),
        ("First time in Morocco and every sense is overwhelmed in the best way 🕌 #Marrakech", 8900, 365),
        ("Remote village with no wifi — most connected I've felt in years 🌿 #DigitalDetox", 7200, 290),
        ("Packing for a 2-week trip, carry-on only challenge! ✈️ #PackingTips", 4500, 310),
        ("Street food in Bangkok for $1 — best pad thai of my life 🍛 #Bangkok", 8100, 295),
        ("Solo travel changed me forever 🌍 Every person should do it at least once. #SoloTravel", 9800, 420),
    ],
    "Food": [
        ("Homemade pasta recipe took 3 attempts to perfect 🍝 SO worth it! #HomeCooking", 6200, 280),
        ("Found the best ramen spot hidden in a tiny alley 🍜 The broth is life-changing. #FoodieFinds", 7800, 310),
        ("Sunday brunch perfection: avocado toast with poached egg 🥑 #BrunchGoals", 8900, 360),
        ("Made croissants from scratch and I'm genuinely so proud! 🥐 #Baking", 9200, 420),
        ("This farm-to-table restaurant changed how I think about food 🌱 #FarmToTable", 5400, 245),
        ("Chocolate lava cake in 20 minutes 🍫 Dessert doesn't have to be complicated! #EasyRecipes", 7100, 290),
        ("Meal prepped for the entire week in 2 hours 🥗 Healthy eating made easy! #MealPrep", 6800, 280),
        ("This vegan cheese board is proof plant-based eating can be incredible 🧀🌿 #Vegan", 5600, 230),
        ("Waited 45 minutes for cold overpriced food 😤 This place is so overrated. #BadFood", 2100, 380),
        ("This recipe was a complete disaster 😞 Wasted ingredients and 2 hours. #CookingFail", 1800, 290),
        ("Three-course dinner from scratch for date night ❤️ Cooking is love. #DateNight", 9800, 450),
        ("Local farmers market haul today 🌽 Supporting small businesses feels amazing. #FarmersMarket", 5100, 195),
        ("Homemade sourdough finally perfected after 6 failed attempts! 🍞 Worth every try. #Sourdough", 8400, 390),
        ("This trending restaurant has amazing aesthetics but terrible food 😒 #Overrated", 3600, 420),
        ("Tried 5 different hot sauces today 🌶️ Ranked them all — #3 is surprising! #HotSauce", 6200, 310),
    ],
    "Fitness": [
        ("New personal record on the squat rack! 💪 6 months of training paying off. #GainsTrain", 7400, 290),
        ("Morning run at sunrise — most underrated way to start your day 🏃 #FitLife", 5600, 210),
        ("Completed my first triathlon! 🏅 What seemed impossible 8 months ago is reality. #Triathlon", 12400, 580),
        ("Yoga practice is transforming my mental state 🧘 Movement is medicine. #Yoga", 6800, 310),
        ("Hit a weight loss milestone today! 🎯 Consistency and patience are everything. #WeightLoss", 9200, 480),
        ("HIIT workout done before 7am ⚡ This discipline is building the life I want. #HIIT", 5100, 195),
        ("Rock climbing for the first time — completely hooked! 🧗 #Climbing", 7300, 280),
        ("Injured my knee again 😢 So frustrated after all the progress I've made. #Injury", 3200, 390),
        ("This fitness program is completely ineffective 😤 Don't waste your money. #FitnessScam", 2800, 450),
        ("Burned out from overtraining this month 😔 Listening to my body now. #Burnout", 3600, 320),
        ("Rest day isn't laziness — it's strategy 😴 Recovery is where real gains happen. #RecoveryDay", 6200, 260),
        ("Swimming 40 laps today 🏊 There's something meditative about being in water. #Swimming", 4800, 185),
        ("Trained through depression and it genuinely saved me 💚 Movement is mental health. #MentalHealth", 11200, 680),
        ("Starting a new 8-week program today 📊 Will document the full journey here. #FitnessJourney", 4100, 195),
        ("Just hit 100 consecutive days of working out! 🔥 Discipline over motivation every time.", 10200, 520),
    ],
    "Tech": [
        ("Just built my first AI model in production 🤖 The future is genuinely wild. #MachineLearning", 5200, 310),
        ("This productivity app completely changed how I manage my time ⚡ #Productivity", 4800, 245),
        ("Open source community came through again 🙌 Built something incredible with free tools. #OpenSource", 3900, 280),
        ("Launched my first mobile app today! 🚀 8 months of nights and weekends shipped. #AppLaunch", 8900, 450),
        ("The M4 chip performance benchmarks are mind-blowing 💻 Apple engineering is insane. #Apple", 6700, 310),
        ("Finally understand neural networks after this course 🧠 The math clicked! #DeepLearning", 4100, 290),
        ("Automated my morning routine with Python scripts 🐍 30 minutes saved daily! #Automation", 5800, 320),
        ("Another major data breach affecting millions 😡 Privacy in tech is broken. #DataPrivacy", 7200, 580),
        ("This framework has the worst documentation I've ever seen 😤 Developers deserve better.", 3100, 390),
        ("6 hours debugging and it was a missing semicolon 💀 I need a career change. #DebuggingHell", 9800, 620),
        ("Big tech layoffs are getting scary 😔 The industry needs to be more humane. #TechLayoffs", 5400, 480),
        ("React vs Vue — both have solid arguments in 2025. What do you use? 💭 #WebDev", 4200, 520),
        ("Setting up a home lab this weekend 🖥️ Will document the entire build process. #HomeLab", 3600, 195),
        ("GitHub Copilot update is incredible 👨‍💻 AI pair programming is the real deal now. #AI", 6200, 340),
        ("Finally switched from Windows to Linux and I'm never going back 🐧 #Linux", 7800, 480),
    ],
    "Lifestyle": [
        ("Journaling every morning for 90 days changed my life 📓 Didn't expect this. #JournalYourLife", 8200, 380),
        ("Decluttered my apartment — mental clarity followed immediately ✨ #Minimalism", 6800, 290),
        ("Planted my first balcony garden 🌱 Growing your own food hits differently. #UrbanGarden", 5400, 240),
        ("Read 3 books this month by swapping 1 hour of scrolling per day 📚 #ReadingChallenge", 7100, 310),
        ("Morning routine overhaul: 5am, no phone, tea, journal 🌅 Life-changing. #MorningRoutine", 9400, 420),
        ("Adopted a rescue dog last month 🐕 The joy this animal brings is indescribable. #DogMom", 14200, 680),
        ("Financial freedom update: debt-free after 3 years of intentional living! 💰 #FinancialFreedom", 10800, 580),
        ("Burnout hit me hard this month 😔 Hustle culture is genuinely toxic. #Burnout", 5600, 490),
        ("This self-help book is just rehashed common sense 😤 Save your money.", 2800, 320),
        ("Social comparison is destroying my mental health 😢 Need to step back. #SocialMedia", 4200, 560),
        ("Digital detox weekend was the reset my brain needed 📵 Try it for 48 hours. #DigitalWellness", 6100, 280),
        ("Volunteered at a local shelter today ❤️ Giving back fills your soul. #GivingBack", 9200, 410),
        ("Learned to say no this year — transformed my relationships 🙅 #Boundaries", 8600, 490),
        ("6 months alcohol-free and I feel incredible 🌿 Best decision I've ever made. #SoberLife", 11200, 620),
        ("Slow Sunday reset: candles, herbal tea, good book 📖 This is what life is for. ✨", 7800, 340),
    ],
}

def seed():
    random.seed(42)
    base_date = datetime.date.today()
    conn = get_conn()

    # Check if already seeded
    existing = conn.execute("SELECT COUNT(*) FROM users WHERE is_demo=1").fetchone()[0]
    if existing >= 10:
        print(f"✅ Demo accounts already exist ({existing} found). Nothing to do.")
        print("\n📋 Demo Login Credentials:")
        print("   All passwords: demo1234")
        for p in DEMO_PROFILES:
            print(f"   Username: {p['username']:25s} | Specialty: {p['specialty']}")
        conn.close()
        return

    print("🌱 Seeding 10 demo profiles...\n")

    for profile in DEMO_PROFILES:
        # Delete if partial
        old = conn.execute("SELECT id FROM users WHERE username=?", (profile["username"],)).fetchone()
        if old:
            conn.execute("DELETE FROM user_posts WHERE user_id=?", (old["id"],))
            conn.execute("DELETE FROM users WHERE id=?", (old["id"],))
            conn.commit()

        # Insert user
        conn.execute(
            "INSERT INTO users (username, email, password_hash, instagram_handle, is_demo) VALUES (?,?,?,?,?)",
            (profile["username"], profile["email"], hash_password(profile["password"]),
             profile["handle"], 1)
        )
        conn.commit()
        uid = conn.execute("SELECT id FROM users WHERE username=?", (profile["username"],)).fetchone()["id"]

        specialty    = profile["specialty"]
        all_cats     = list(DEMO_POSTS.keys())
        other_cats   = [c for c in all_cats if c != specialty]
        posts_to_add = []

        # 15 primary specialty posts
        for caption, base_likes, base_comments in DEMO_POSTS[specialty]:
            likes    = int(base_likes    * random.uniform(0.85, 1.15))
            comments = int(base_comments * random.uniform(0.85, 1.15))
            days_ago = random.randint(1, 180)
            post_date = str(base_date - datetime.timedelta(days=days_ago))
            label, score = analyze_sentiment(caption)
            eng = likes + 2 * comments
            posts_to_add.append((uid, caption, specialty, likes, comments, post_date, label, score, eng))

        # 6 cross-category posts (3 from each of 2 other categories)
        random.shuffle(other_cats)
        for extra_cat in other_cats[:2]:
            for caption, base_likes, base_comments in random.sample(DEMO_POSTS[extra_cat], 3):
                likes    = int(base_likes    * random.uniform(0.7, 1.0))
                comments = int(base_comments * random.uniform(0.7, 1.0))
                days_ago = random.randint(1, 180)
                post_date = str(base_date - datetime.timedelta(days=days_ago))
                label, score = analyze_sentiment(caption)
                eng = likes + 2 * comments
                posts_to_add.append((uid, caption, extra_cat, likes, comments, post_date, label, score, eng))

        conn.executemany("""INSERT INTO user_posts
            (user_id,caption,category,likes,comments,post_date,sentiment_label,sentiment_score,engagement_score)
            VALUES (?,?,?,?,?,?,?,?,?)""", posts_to_add)
        conn.commit()

        print(f"  ✅ @{profile['handle']:20s} | {len(posts_to_add):2d} posts | Specialty: {specialty}")

    conn.close()

    print(f"\n🎉 Done! 10 demo profiles created.\n")
    print("━" * 55)
    print("📋 DEMO LOGIN CREDENTIALS")
    print("━" * 55)
    print(f"{'Username':<25} {'Password':<12} {'Specialty'}")
    print("─" * 55)
    for p in DEMO_PROFILES:
        print(f"{p['username']:<25} {p['password']:<12} {p['specialty']}")
    print("━" * 55)
    print("\n👉 Now run: streamlit run app.py")

if __name__ == "__main__":
    init_db()
    seed()