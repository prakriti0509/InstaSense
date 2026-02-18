"""
InstaSense — AI Instagram Analytics Dashboard v3.0
- User auth (register/login)
- Emoji picker on caption input
- Emoji-aware sentiment analysis
- 10 demo profiles with ~15 posts each (500 demo posts total)
- Real analytics & recommendations per profile
Run: streamlit run app.py
"""

import os, sys, hashlib, sqlite3, datetime, random, re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="InstaSense — AI Analytics", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"]{background-color:#0A0A0F!important;color:#E8E8F0!important;font-family:'DM Sans',sans-serif!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0F0F1A 0%,#0A0A13 100%)!important;border-right:1px solid #1E1E2E!important}
[data-testid="stSidebar"] *{color:#C8C8D8!important}

.auth-card{background:linear-gradient(135deg,#13131F,#0F0F1A);border:1px solid #1E1E30;border-radius:20px;padding:40px 48px;max-width:480px;margin:40px auto}
.auth-title{font-size:26px;font-weight:700;color:#F0F0FF;margin-bottom:6px;text-align:center}
.auth-sub{font-size:13px;color:#5858A8;text-align:center;margin-bottom:28px}

.kpi-card{background:linear-gradient(135deg,#13131F 0%,#0F0F1A 100%);border:1px solid #1E1E30;border-radius:16px;padding:24px;position:relative;overflow:hidden;transition:transform 0.2s ease}
.kpi-card:hover{transform:translateY(-2px);border-color:#2E2E48}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent,linear-gradient(90deg,#8B5CF6,#6366F1))}
.kpi-icon{font-size:28px;margin-bottom:12px;display:block}
.kpi-value{font-size:32px;font-weight:700;color:#F0F0FF;line-height:1;letter-spacing:-1px;margin-bottom:4px}
.kpi-label{font-size:12px;font-weight:500;color:#6868A8;text-transform:uppercase;letter-spacing:1.5px}
.kpi-sub{font-size:12px;color:#4CAF82;margin-top:8px;font-weight:500}

.section-title{font-size:13px;font-weight:600;color:#4848A8;text-transform:uppercase;letter-spacing:2px;margin:28px 0 16px 0}
.page-header{padding:0 0 24px 0;border-bottom:1px solid #1E1E30;margin-bottom:32px}
.page-title{font-size:28px;font-weight:700;color:#F0F0FF;letter-spacing:-0.5px;margin:0}
.page-subtitle{font-size:14px;color:#5858A8;margin:4px 0 0 0}
.rec-card{background:#0F0F1A;border:1px solid #1E1E30;border-left:4px solid #8B5CF6;border-radius:10px;padding:16px 20px;margin:8px 0;font-size:14px;line-height:1.6;color:#B8B8D8}

.stButton>button{background:linear-gradient(135deg,#8B5CF6,#6366F1)!important;color:white!important;border:none!important;border-radius:10px!important;padding:12px 28px!important;font-family:'DM Sans',sans-serif!important;font-weight:600!important;font-size:14px!important;transition:all 0.2s ease!important;box-shadow:0 4px 15px rgba(139,92,246,0.3)!important}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 20px rgba(139,92,246,0.45)!important}

.stTextInput>div>div>input,.stSelectbox>div>div,.stNumberInput>div>div>input,.stTextArea>div>div>textarea{background:#0F0F1A!important;border:1px solid #1E1E30!important;border-radius:10px!important;color:#E8E8F0!important;font-family:'DM Sans',sans-serif!important}

.logo-area{padding:20px 16px 28px 16px;border-bottom:1px solid #1E1E2A;margin-bottom:16px}
.logo-text{font-size:22px;font-weight:700;color:#E0E0FF;letter-spacing:-0.5px}
.logo-tagline{font-size:11px;color:#4848A8;letter-spacing:1.5px;text-transform:uppercase;margin-top:2px}

.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:0.5px;text-transform:uppercase}
.badge-positive{background:rgba(76,175,130,0.15);color:#4CAF82}
.badge-negative{background:rgba(239,83,80,0.15);color:#EF5350}
.badge-demo{background:rgba(245,158,11,0.15);color:#F59E0B}

.empty-state{text-align:center;padding:60px 20px;color:#4848A8}
.empty-state-icon{font-size:48px;margin-bottom:16px}
.empty-state-title{font-size:20px;font-weight:600;color:#8888B8;margin-bottom:8px}
.empty-state-sub{font-size:14px;color:#4848A8}

/* Emoji picker */
.emoji-grid{display:flex;flex-wrap:wrap;gap:6px;padding:12px;background:#0F0F1A;border:1px solid #1E1E30;border-radius:12px;max-height:220px;overflow-y:auto}
.emoji-btn{font-size:22px;cursor:pointer;padding:4px 6px;border-radius:6px;border:none;background:transparent;transition:background 0.15s}
.emoji-btn:hover{background:rgba(139,92,246,0.15)}

#MainMenu{visibility:hidden}footer{visibility:hidden}[data-testid="stDecoration"]{display:none}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#9898C8", size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor="#1A1A2E", linecolor="#1A1A2E", tickcolor="#4848A8"),
    yaxis=dict(gridcolor="#1A1A2E", linecolor="#1A1A2E", tickcolor="#4848A8"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9898C8")),
)
CATEGORY_COLORS = {
    "Fashion":"#8B5CF6","Travel":"#6366F1","Food":"#EC4899",
    "Fitness":"#14B8A6","Tech":"#F59E0B","Lifestyle":"#10B981",
}

# ══════════════════════════════════════════════════════════════════════════════
# EMOJI DATA
# ══════════════════════════════════════════════════════════════════════════════
EMOJI_CATEGORIES = {
    "😊 Faces": ["😀","😃","😄","😁","😆","😅","😂","🤣","😊","😇","🥰","😍","🤩","😘","😗","😚","😙","🥲","😋","😛","😜","🤪","😝","🤑","🤗","🤭","🤫","🤔","😐","😑","😶","😏","😒","🙄","😬","🤥","😌","😔","😪","🤤","😴","😷","🤒","🤕","🤢","🤮","🤧","🥵","🥶","🥴","😵","🤯","🤠","🥳","🥸","😎","🤓","🧐","😕","😟","🙁","☹️","😮","😯","😲","😳","🥺","😦","😧","😨","😰","😥","😢","😭","😱","😖","😣","😞","😓","😩","😫","🥱","😤","😡","😠","🤬","😈","👿","💀","☠️","💩","🤡","👹","👺","👻","👽","👾","🤖"],
    "❤️ Hearts": ["❤️","🧡","💛","💚","💙","💜","🖤","🤍","🤎","💔","❣️","💕","💞","💓","💗","💖","💘","💝","💟","☮️","✝️","☪️","🕉️","☸️","✡️","🔯","🕎","☯️","☦️","🛐","⛎","♈","♉","♊","♋","♌","♍","♎","♏","♐","♑","♒","♓","🆔","⚛️","🉑","☢️","☣️","📴","📳","🈶","🈚","🈸","🈺","🈷️","✴️","🆚","💮","🉐","㊙️","㊗️","🈴","🈵","🈹","🈲","🅰️","🅱️","🆎","🆑","🅾️","🆘","❌","⭕","🛑","⛔","📛","🚫","💯","💢","♨️","🚷","🚯","🚳","🚱","🔞","📵","🚭","❗","❕","❓","❔","‼️","⁉️","🔅","🔆","〽️","⚠️","🚸","🔱","⚜️","🔰","♻️","✅","🈯","💹","❎","🌐","💠","Ⓜ️","🌀","💤","🏧","🚾","♿","🅿️","🈳","🈹"],
    "🔥 Popular": ["🔥","✨","💫","⭐","🌟","💥","🎉","🎊","🙌","👏","💪","🚀","💯","🏆","👑","💎","🌈","🦋","🌸","🌺","🍀","🎯","❤️‍🔥","🥳","😍","🤩","💃","🕺","🎶","🎵","🌙","☀️","🌊","⚡","🦁","🐯","🦊","🐺","🦄","🐉","🌹","🍁","🎭","🎪","🎠","🎡","🎢","🎆","🎇","🧨","✨","🎈","🎀","🎁","🎗️","🎟️","🎫"],
    "👍 Gestures": ["👍","👎","👌","🤌","🤏","✌️","🤞","🤟","🤘","🤙","👈","👉","👆","🖕","👇","☝️","👋","🤚","🖐️","✋","🖖","👏","🙌","🤲","🤝","🙏","✍️","💅","🤳","💪","🦾","🦿","🦵","🦶","👂","🦻","👃","🧠","🫀","🫁","🦷","🦴","👀","👁️","👅","👄","💋","🩸"],
    "📸 Lifestyle": ["📸","📷","🎥","📱","💻","🖥️","⌨️","🖱️","🎮","🕹️","📺","📻","📡","🔭","🔬","💡","🔦","🕯️","🪔","🧯","🛢️","💰","💳","💎","⚖️","🧲","🔧","🔨","⚒️","🛠️","⛏️","🔩","🪛","🔫","🪃","🏹","🛡️","🪚","🔪","🗡️","⚔️","🛠️","🪝","🪜","🧱","🪞","🚪","🛋️","🪑","🚽","🪠","🚿","🛁","🪤","🧴","🧷","🧹","🧺","🧻","🧼","🫧","🪣","🧽","🪒","🧴","🪥","🧹","🧺"],
    "🍕 Food": ["🍕","🍔","🌮","🌯","🥗","🍜","🍝","🍛","🍲","🥘","🍱","🍣","🍤","🦞","🦀","🦑","🍦","🍧","🍨","🍩","🍪","🎂","🍰","🧁","🥧","🍫","🍬","🍭","🍮","🍯","🍷","🍸","🍹","🧃","☕","🍵","🧋","🥤","🍺","🍻","🥂","🥃","🧊","🥛","🍼","🫖","🧉","🥃","🍾"],
    "✈️ Travel": ["✈️","🚀","🛸","🚁","🛩️","🛫","🛬","🪂","💺","🚂","🚃","🚄","🚅","🚆","🚇","🚈","🚉","🚊","🚞","🚝","🚋","🚌","🚍","🚎","🚐","🚑","🚒","🚓","🚔","🚕","🚖","🚗","🚘","🚙","🛻","🚚","🚛","🚜","🏎️","🏍️","🛵","🛺","🚲","🛴","🛹","🛼","🚏","🛣️","🛤️","⛽","🚧","⚓","🛟","⛵","🚤","🛥️","🛳️","⛴️","🚢","🗺️","🧭","🌍","🌎","🌏","🏔️","⛰️","🗻","🏕️","🏖️","🏜️","🏝️","🏞️","🏟️","🏛️","🏗️","🧱","🏘️","🏚️","🏠","🏡","🏢","🏣","🏤","🏥","🏦","🏨","🏩","🏪","🏫","🏬","🏭","🏯","🏰","💒","🗼","🗽","⛪","🕌","🕍","🛕","⛩️","🕋","⛲","⛺","🌁","🌃","🏙️","🌄","🌅","🌆","🌇","🌉","🌌","🎠","🎡","🎢","💈","🎪"],
    "💪 Fitness": ["💪","🏋️","🤸","🧘","🏊","🚴","🏃","🚶","🧗","🏇","⛷️","🏂","🪂","🏄","🚣","🧜","🏌️","🏹","🥊","🥋","🏆","🥇","🥈","🥉","🏅","🎖️","🎗️","🏵️","🎫","🎟️","🤺","🏇","⛷️","🏂","🪂","🏋️","🤼","🤸","⛹️","🤺","🏊","🚴","🏄","🛶","🎽","🥅","⛳","🎣","🤿","🎽","🎿","🛷","🥌","🎯","🪃","🎱","🔮","🎮","🎲","🎰","🧩","🎭","🎨"],
}

# Emoji sentiment weights for analysis
EMOJI_SENTIMENT = {
    "positive": {"😍":0.9,"🔥":0.8,"💯":0.8,"✨":0.7,"🌟":0.8,"💫":0.7,"🎉":0.8,"🙌":0.7,
                 "💪":0.7,"🚀":0.8,"🏆":0.8,"👑":0.8,"💎":0.7,"🌈":0.7,"❤️":0.8,"😊":0.7,
                 "😁":0.7,"🤩":0.9,"🥰":0.9,"😘":0.8,"👍":0.7,"💚":0.7,"💙":0.7,"💜":0.7,
                 "🎊":0.8,"💃":0.7,"🕺":0.7,"⭐":0.7,"🌸":0.6,"🍀":0.6,"😄":0.7,"😃":0.7,
                 "🥳":0.8,"❤️‍🔥":0.9,"💖":0.8,"💕":0.7,"🫶":0.8},
    "negative": {"😤":0.8,"😡":0.9,"😠":0.8,"🤬":0.9,"😞":0.8,"😔":0.7,"😢":0.8,"😭":0.8,
                 "😤":0.7,"💀":0.7,"👎":0.8,"😒":0.7,"🙄":0.6,"😩":0.7,"😫":0.7,"😖":0.8,
                 "😣":0.8,"😨":0.7,"😰":0.8,"😱":0.7,"🤮":0.9,"😷":0.5,"☹️":0.7,"🥺":0.6},
}

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "instasense_users.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_user_db():
    conn = get_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, instagram_handle TEXT,
        is_demo INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS user_posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL, caption TEXT, category TEXT,
        likes INTEGER DEFAULT 0, comments INTEGER DEFAULT 0,
        post_date TEXT, sentiment_label TEXT, sentiment_score REAL,
        engagement_score REAL, created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id))""")
    conn.commit()
    conn.close()

def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def register_user(username, email, password, instagram_handle, is_demo=0):
    try:
        conn = get_conn()
        conn.execute("INSERT INTO users (username,email,password_hash,instagram_handle,is_demo) VALUES (?,?,?,?,?)",
            (username.strip(), email.strip().lower(), hash_password(password), instagram_handle.strip(), is_demo))
        conn.commit()
        uid = conn.execute("SELECT id FROM users WHERE username=?", (username.strip(),)).fetchone()["id"]
        conn.close()
        return True, "Account created!", uid
    except sqlite3.IntegrityError as e:
        return False, ("Username already taken." if "username" in str(e) else "Email already registered."), None
    except Exception as e:
        return False, f"Error: {e}", None

def login_user(username, password):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=? AND password_hash=?",
        (username.strip(), hash_password(password))).fetchone()
    conn.close()
    return (True, dict(row)) if row else (False, None)

def add_post(user_id, caption, category, likes, comments, post_date):
    label, score = analyze_sentiment(caption)
    eng = likes + (2 * comments)
    conn = get_conn()
    conn.execute("""INSERT INTO user_posts
        (user_id,caption,category,likes,comments,post_date,sentiment_label,sentiment_score,engagement_score)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (user_id, caption, category, likes, comments, post_date, label, score, eng))
    conn.commit()
    conn.close()
    return label

def bulk_insert_posts(posts: list):
    conn = get_conn()
    conn.executemany("""INSERT INTO user_posts
        (user_id,caption,category,likes,comments,post_date,sentiment_label,sentiment_score,engagement_score)
        VALUES (?,?,?,?,?,?,?,?,?)""", posts)
    conn.commit()
    conn.close()

def get_user_posts(user_id):
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM user_posts WHERE user_id=? ORDER BY post_date DESC", conn, params=(user_id,))
    conn.close()
    return df

def delete_post(post_id, user_id):
    conn = get_conn()
    conn.execute("DELETE FROM user_posts WHERE id=? AND user_id=?", (post_id, user_id))
    conn.commit()
    conn.close()

def demo_users_exist():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM users WHERE is_demo=1").fetchone()[0]
    conn.close()
    return n >= 10

# ══════════════════════════════════════════════════════════════════════════════
# EMOJI-AWARE SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
def analyze_sentiment(text: str):
    if not text or not text.strip():
        return "Neutral", 0.5
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        # Extract emoji sentiment boost before stripping
        emoji_boost = 0.0
        for emoji, weight in EMOJI_SENTIMENT["positive"].items():
            if emoji in text:
                emoji_boost += weight * text.count(emoji) * 0.12
        for emoji, weight in EMOJI_SENTIMENT["negative"].items():
            if emoji in text:
                emoji_boost -= weight * text.count(emoji) * 0.12
        emoji_boost = max(-0.6, min(0.6, emoji_boost))

        # Clean text for VADER
        clean = re.sub(r"http\S+|@\w+", "", text)
        clean = re.sub(r"#(\w+)", r"\1", clean)
        clean = clean.encode("ascii", errors="ignore").decode("ascii").strip()

        POS = {"great","obsessed","amazing","love","best","perfect","incredible","wonderful",
               "fantastic","awesome","beautiful","proud","milestone","achieved","joy","grateful",
               "delicious","stunning","worth","magical","excited","happy","blessed","thrilled"}
        NEG = {"terrible","worst","hate","disappointed","frustrating","broken","overpriced",
               "disaster","wasted","scam","ruined","failed","toxic","burnout","injured","setback",
               "overrated","miserable","awful","horrible","dreadful","annoying","useless"}

        analyzer = SentimentIntensityAnalyzer()
        scores   = analyzer.polarity_scores(clean)
        compound = scores["compound"]
        words    = set(clean.lower().split())
        compound = max(-1.0, min(1.0, compound + len(words&POS)*0.08 - len(words&NEG)*0.08 + emoji_boost))

        if compound >= 0.03:
            return "Positive", round(min(0.5 + compound/2, 0.99), 4)
        else:
            return "Negative", round(min(0.5 + abs(compound)/2, 0.99), 4)
    except Exception:
        return "Neutral", 0.5

# ══════════════════════════════════════════════════════════════════════════════
# DEMO DATA SEEDING — 10 profiles × ~15 posts each
# ══════════════════════════════════════════════════════════════════════════════
DEMO_PROFILES = [
    {"username":"fashionista_aria",  "password":"demo1234", "email":"aria@demo.com",    "handle":"@aria_styles",    "specialty":"Fashion"},
    {"username":"wanderlust_kai",    "password":"demo1234", "email":"kai@demo.com",     "handle":"@kai_travels",    "specialty":"Travel"},
    {"username":"foodie_priya",      "password":"demo1234", "email":"priya@demo.com",   "handle":"@priya_eats",     "specialty":"Food"},
    {"username":"fitlife_marcus",    "password":"demo1234", "email":"marcus@demo.com",  "handle":"@marcus_fits",    "specialty":"Fitness"},
    {"username":"techbro_sam",       "password":"demo1234", "email":"sam@demo.com",     "handle":"@sam_codes",      "specialty":"Tech"},
    {"username":"lifestyle_zoe",     "password":"demo1234", "email":"zoe@demo.com",     "handle":"@zoe_vibes",      "specialty":"Lifestyle"},
    {"username":"glamour_nina",      "password":"demo1234", "email":"nina@demo.com",    "handle":"@nina_glam",      "specialty":"Fashion"},
    {"username":"explorer_raj",      "password":"demo1234", "email":"raj@demo.com",     "handle":"@raj_explores",   "specialty":"Travel"},
    {"username":"chef_isabella",     "password":"demo1234", "email":"isabella@demo.com","handle":"@bella_cooks",    "specialty":"Food"},
    {"username":"wellness_leo",      "password":"demo1234", "email":"leo@demo.com",     "handle":"@leo_wellness",   "specialty":"Fitness"},
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
        ("Tried to recreate a trending look but it just didn't work out 😔 Frustrating.", 3100, 195),
        ("New season wardrobe staples 🎀 These pieces will never go out of style. #TimelessFashion", 6700, 260),
        ("Capsule wardrobe update complete! Less is truly more. 🌿 #CapsuleWardrobe", 5900, 230),
        ("Fast fashion waste is getting out of hand 🌍 This industry needs a reality check. #SlowFashion", 4100, 410),
        ("Styling tips for petite frames — your body is perfect as it is! ❤️ #BodyPositive", 8800, 520),
    ],
    "Travel": [
        ("Woke up to this view and forgot all my worries 🌅 Travel heals everything. #Wanderlust", 9200, 340),
        ("Hidden gem in Bali was the most magical place I've ever visited! 🌺 #Bali", 11500, 480),
        ("Road trip complete! 2000 miles and countless memories 🚗 #RoadTrip", 7800, 295),
        ("Lost in the streets of Tokyo 🗼 This city has my whole heart. #Tokyo", 10200, 410),
        ("Santorini sunsets hit differently 🌇 Put this on your bucket list NOW. #Greece", 13400, 560),
        ("Explored an ancient temple today — 2000 years old 🏛️ Perspective is everything. #History", 6100, 245),
        ("The locals here made us feel so welcome 🤝 Travel teaches kindness. #Culture", 5400, 310),
        ("This travel company completely ruined our vacation. Zero stars. 😡 #BadExperience", 3800, 490),
        ("Overtourism has destroyed this once-beautiful destination 😔 So disappointing. #Overtourism", 4200, 380),
        ("Missed my connecting flight due to airline mismanagement 😤 Worst travel day ever.", 2900, 340),
        ("Paragliding over the Alps was the best decision of my life 🪂 #Alps", 12100, 520),
        ("First time in Morocco and every sense is overwhelmed in the best way 🕌 #Marrakech", 8900, 365),
        ("Remote village with no wifi — most connected I've felt in years 🌿 #DigitalDetox", 7200, 290),
        ("Packing for a 2-week trip, carry-on only challenge! ✈️ #PackingTips", 4500, 310),
        ("Street food in Bangkok for $1 — best pad thai of my life 🍛 #Bangkok", 8100, 295),
    ],
    "Food": [
        ("Homemade pasta recipe took 3 attempts to perfect 🍝 SO worth it! #HomeCooking", 6200, 280),
        ("Found the best ramen spot hidden in a tiny alley 🍜 The broth is life-changing. #FoodieFinds", 7800, 310),
        ("Sunday brunch perfection: avocado toast with poached egg on sourdough 🥑 #BrunchGoals", 8900, 360),
        ("Made croissants from scratch and I'm genuinely so proud! 🥐 #Baking", 9200, 420),
        ("This farm-to-table restaurant changed how I think about food 🌱 #FarmToTable", 5400, 245),
        ("Chocolate lava cake in 20 minutes 🍫 Dessert doesn't have to be complicated! #EasyRecipes", 7100, 290),
        ("Meal prepped for the entire week in 2 hours 🥗 Healthy eating made easy! #MealPrep", 6800, 280),
        ("This vegan cheese board is proof plant-based eating can be incredible 🧀🌿 #Vegan", 5600, 230),
        ("Waited 45 minutes for cold overpriced food 😤 This place is so overrated. #BadFood", 2100, 380),
        ("This recipe was a complete disaster 😞 Wasted ingredients and 2 hours. #CookingFail", 1800, 290),
        ("Food delivery took 90 minutes and arrived completely wrong 😡 So frustrated.", 2400, 340),
        ("Trying out a new recipe this weekend — will share results on Sunday! 🍳 #CookingAdventures", 3200, 150),
        ("Three-course dinner made from scratch for date night ❤️ Cooking is love. #DateNight", 9800, 450),
        ("Local farmers market haul today 🌽 Supporting small businesses feels amazing. #FarmersMarket", 5100, 195),
        ("This trending restaurant has amazing aesthetics but terrible food 😒 Style over substance. 👎", 3600, 420),
    ],
    "Fitness": [
        ("New personal record on the squat rack! 💪 6 months of training paying off. #GainsTrain", 7400, 290),
        ("Morning run at sunrise — most underrated way to start your day 🏃 #FitLife", 5600, 210),
        ("Completed my first triathlon! 🏅 What seemed impossible 8 months ago is reality. #Triathlon", 12400, 580),
        ("Yoga practice is transforming my mental state 🧘 Movement is medicine. #Yoga", 6800, 310),
        ("Hit a weight loss milestone today! 🎯 Consistency and patience are everything. #WeightLoss", 9200, 480),
        ("HIIT workout done before 7am ⚡ This discipline is building the life I want. #HIIT", 5100, 195),
        ("Rock climbing for the first time — completely hooked! 🧗 Upper body workout is no joke. #Climbing", 7300, 280),
        ("Injured my knee again 😢 So frustrated after all the progress I've made. #Injury", 3200, 390),
        ("This fitness influencer's program is completely ineffective 😤 Don't waste your money. #FitnessScam", 2800, 450),
        ("Burned out from overtraining this month 😔 Listening to my body now. #Burnout", 3600, 320),
        ("Rest day isn't laziness — it's strategy 😴 Recovery is where real gains happen. #RecoveryDay", 6200, 260),
        ("Swimming 40 laps today 🏊 There's something meditative about being in water. #Swimming", 4800, 185),
        ("Trained through depression and it genuinely saved me 💚 Movement is mental health. #MentalHealth", 11200, 680),
        ("Starting a new 8-week program today 📊 Will document the full journey here. #FitnessJourney", 4100, 195),
        ("New gym gear arrived 🎽 Whether gear matters is debatable but I feel ready! #GymWear", 3800, 145),
    ],
    "Tech": [
        ("Just built my first AI model in production 🤖 The future is genuinely wild. #MachineLearning", 5200, 310),
        ("This productivity app completely changed how I manage my time ⚡ #Productivity", 4800, 245),
        ("Open source community came through again 🙌 Built something incredible with free tools. #OpenSource", 3900, 280),
        ("Launched my first mobile app today! 🚀 8 months of nights and weekends shipped. #AppLaunch", 8900, 450),
        ("The M4 chip performance benchmarks are mind-blowing 💻 Apple engineering is insane. #Apple", 6700, 310),
        ("Finally understand neural networks after this course 🧠 The math clicked overnight. #DeepLearning", 4100, 290),
        ("Automated my morning routine with Python scripts 🐍 30 minutes saved daily! #Automation", 5800, 320),
        ("Another major data breach affecting millions 😡 Privacy in tech is completely broken. #DataPrivacy", 7200, 580),
        ("This framework has the worst documentation I've ever seen 😤 Developers deserve better. #DevExperience", 3100, 390),
        ("6 hours debugging and it was a missing semicolon 💀 I need a career change. #DebuggingHell", 9800, 620),
        ("Big tech layoffs are getting scary 😔 The industry needs to be more humane. #TechLayoffs", 5400, 480),
        ("React vs Vue for a new project — both have solid arguments. What do you use? 💭 #WebDev", 4200, 520),
        ("Setting up a home lab this weekend 🖥️ Will document the entire build process. #HomeLab", 3600, 195),
        ("WebAssembly is opening a new era of browser apps 🌐 This is genuinely exciting. #WASM", 3100, 210),
        ("GitHub Copilot update is incredible 👨‍💻 AI pair programming is the real deal now. #AI", 6200, 340),
    ],
    "Lifestyle": [
        ("Journaling every morning for 90 days changed my life 📓 Didn't expect this. #JournalYourLife", 8200, 380),
        ("Decluttered my apartment — mental clarity followed immediately ✨ #Minimalism", 6800, 290),
        ("Planted my first balcony garden 🌱 Growing your own food hits differently. #UrbanGarden", 5400, 240),
        ("Read 3 books this month by swapping 1 hour of scrolling per day 📚 Trade-offs matter. #ReadingChallenge", 7100, 310),
        ("Morning routine overhaul complete: 5am, no phone, tea, journal 🌅 Life-changing. #MorningRoutine", 9400, 420),
        ("Adopted a rescue dog last month 🐕 The joy this animal brings is indescribable. #DogMom", 14200, 680),
        ("Financial freedom update: debt-free after 3 years of intentional living! 💰 #FinancialFreedom", 10800, 580),
        ("Burnout hit me hard this month 😔 Hustle culture is genuinely toxic. #Burnout", 5600, 490),
        ("This self-help book is just rehashed common sense 😤 Save your money. #Overrated", 2800, 320),
        ("Work-life balance is a myth when building something 😞 Something always suffers.", 3400, 280),
        ("Social comparison is destroying my mental health 😢 Need to step back. #SocialMedia", 4200, 560),
        ("Digital detox weekend was the reset my brain needed 📵 Try it for 48 hours. #DigitalWellness", 6100, 280),
        ("Volunteered at a local shelter today ❤️ Giving back fills your soul. #GivingBack", 9200, 410),
        ("Learned to say no this year — transformed my relationships 🙅 #Boundaries", 8600, 490),
        ("New morning routine this week ☕ Will report back whether it's sustainable. #MorningVibes", 3800, 150),
    ],
}

def seed_demo_data():
    """Create 10 demo profiles with ~15 posts each."""
    if demo_users_exist():
        return

    random.seed(42)
    base_date = datetime.date.today()

    for profile in DEMO_PROFILES:
        ok, msg, uid = register_user(
            profile["username"], profile["email"], profile["password"],
            profile["handle"], is_demo=1
        )
        if not ok or uid is None:
            # Try to get existing uid
            conn = get_conn()
            row = conn.execute("SELECT id FROM users WHERE username=?", (profile["username"],)).fetchone()
            conn.close()
            if row:
                uid = row["id"]
            else:
                continue

        specialty = profile["specialty"]
        posts_data = []

        # Primary category posts (specialty)
        primary_posts = DEMO_POSTS[specialty]
        for i, (caption, base_likes, base_comments) in enumerate(primary_posts):
            likes    = int(base_likes    * random.uniform(0.85, 1.15))
            comments = int(base_comments * random.uniform(0.85, 1.15))
            days_ago = random.randint(1, 180)
            post_date = str(base_date - datetime.timedelta(days=days_ago))
            label, score = analyze_sentiment(caption)
            eng = likes + 2 * comments
            posts_data.append((uid, caption, specialty, likes, comments, post_date, label, score, eng))

        # Add 5 posts from 2 other random categories (cross-posting)
        other_cats = [c for c in DEMO_POSTS.keys() if c != specialty]
        random.shuffle(other_cats)
        for extra_cat in other_cats[:2]:
            sample = random.sample(DEMO_POSTS[extra_cat], min(3, len(DEMO_POSTS[extra_cat])))
            for caption, base_likes, base_comments in sample:
                likes    = int(base_likes    * random.uniform(0.7, 1.0))
                comments = int(base_comments * random.uniform(0.7, 1.0))
                days_ago = random.randint(1, 180)
                post_date = str(base_date - datetime.timedelta(days=days_ago))
                label, score = analyze_sentiment(caption)
                eng = likes + 2 * comments
                posts_data.append((uid, caption, extra_cat, likes, comments, post_date, label, score, eng))

        bulk_insert_posts(posts_data)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_kpis(df):
    if df.empty: return {}
    df = df.copy()
    df["ns"] = df.apply(lambda r: r["sentiment_score"] if r["sentiment_label"]=="Positive"
        else (1-r["sentiment_score"] if r["sentiment_label"]=="Negative" else 0.5), axis=1)
    top_cat = df.groupby("category")["engagement_score"].mean().idxmax() if len(df["category"].unique())>0 else "—"
    return {"total_posts":len(df), "avg_engagement":round(df["engagement_score"].mean(),1),
            "avg_sentiment":round(df["ns"].mean()*100,1), "top_category":top_cat,
            "positive_rate":round((df["sentiment_label"]=="Positive").mean()*100,1),
            "total_likes":int(df["likes"].sum()), "total_comments":int(df["comments"].sum())}

def compute_category_summary(df):
    if df.empty or "category" not in df.columns: return pd.DataFrame()
    df = df.copy()
    df["ns"] = df.apply(lambda r: r["sentiment_score"] if r["sentiment_label"]=="Positive" else 1-r["sentiment_score"], axis=1)
    s = df.groupby("category").agg(
        total_posts=("id","count"), avg_likes=("likes","mean"), avg_comments=("comments","mean"),
        avg_engagement=("engagement_score","mean"), avg_sentiment=("ns","mean"),
        positive_count=("sentiment_label", lambda x: (x=="Positive").sum()),
    ).reset_index()
    s["positive_rate"]  = (s["positive_count"]/s["total_posts"]*100).round(1)
    s["avg_likes"]      = s["avg_likes"].round(0).astype(int)
    s["avg_comments"]   = s["avg_comments"].round(0).astype(int)
    s["avg_engagement"] = s["avg_engagement"].round(1)
    e = s["avg_engagement"]; sv = s["avg_sentiment"]
    en = (e-e.min())/(e.max()-e.min()+1e-9)
    sn = (sv-sv.min())/(sv.max()-sv.min()+1e-9)
    s["composite"] = (0.6*en+0.4*sn).round(4)
    s["rank"] = s["composite"].rank(ascending=False).astype(int)
    return s.sort_values("rank").drop(columns=["positive_count"])

def generate_recs(cat_df, total_posts, handle):
    if cat_df.empty: return ["Add posts across different categories to unlock recommendations."]
    recs = []
    top = cat_df.iloc[0]; bottom = cat_df.iloc[-1]
    recs.append(f"🏆 **{handle}** should focus on **{top['category']}** — top category with avg engagement {top['avg_engagement']:,.0f} and {top['positive_rate']}% positive posts.")
    if len(cat_df) > 2:
        hs = cat_df[(cat_df["avg_sentiment"]>0.70)&(cat_df["rank"]>2)]
        if not hs.empty:
            recs.append(f"💡 **{hs.iloc[0]['category']}** has great sentiment but lower reach — boost with hashtags and collabs.")
        ls = cat_df[(cat_df["avg_sentiment"]<0.65)&(cat_df["rank"]<=3)&(cat_df["category"]!=top["category"])]
        if not ls.empty:
            recs.append(f"⚠️ **{ls.iloc[0]['category']}** gets engagement but mixed sentiment — refine your tone and reply to comments.")
    recs.append(f"📉 **{bottom['category']}** is your weakest — try new formats, different posting times, or a new content angle.")
    if total_posts < 10:
        recs.append(f"📊 Only {total_posts} posts tracked — add more for increasingly accurate and meaningful insights.")
    recs.append("⏰ Post at 7–9am or 7–10pm local time for maximum initial engagement velocity.")
    return recs

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE + INIT
# ══════════════════════════════════════════════════════════════════════════════
init_user_db()
seed_demo_data()

for k, v in [("logged_in",False),("user",None),("page","Home"),("auth_mode","login"),("caption_text",""),("emoji_tab","🔥 Popular")]:
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGES
# ══════════════════════════════════════════════════════════════════════════════
def show_login():
    st.markdown("""<div style="text-align:center;padding:40px 0 20px 0;">
        <div style="font-size:40px;">📊</div>
        <div style="font-size:32px;font-weight:700;color:#F0F0FF;letter-spacing:-1px;">InstaSense</div>
        <div style="font-size:13px;color:#4848A8;letter-spacing:2px;text-transform:uppercase;margin-top:4px;">AI Instagram Analytics</div>
    </div>""", unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown('<div class="auth-title">Welcome back</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Sign in to your analytics dashboard</div>', unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="your_username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")

        if st.button("Sign In", use_container_width=True):
            if not username or not password:
                st.error("Please fill in all fields.")
            else:
                ok, user = login_user(username, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        # Demo accounts box

        st.markdown('<div style="text-align:center;color:#5858A8;font-size:13px;margin-top:8px;">Don\'t have an account?</div>', unsafe_allow_html=True)
        if st.button("Create Account →", use_container_width=True, key="goto_register"):
            st.session_state.auth_mode = "register"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def show_register():
    st.markdown("""<div style="text-align:center;padding:40px 0 20px 0;">
        <div style="font-size:40px;">📊</div>
        <div style="font-size:32px;font-weight:700;color:#F0F0FF;letter-spacing:-1px;">InstaSense</div>
        <div style="font-size:13px;color:#4848A8;letter-spacing:2px;text-transform:uppercase;margin-top:4px;">Create Your Account</div>
    </div>""", unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown('<div class="auth-title">Create account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Start analyzing your Instagram performance</div>', unsafe_allow_html=True)

        username  = st.text_input("Username",         placeholder="coolcreator",    key="reg_user")
        email     = st.text_input("Email",            placeholder="you@example.com",key="reg_email")
        insta     = st.text_input("Instagram Handle", placeholder="@yourhandle",    key="reg_insta")
        password  = st.text_input("Password",         type="password", placeholder="••••••••", key="reg_pass")
        password2 = st.text_input("Confirm Password", type="password", placeholder="••••••••", key="reg_pass2")

        if st.button("Create Account", use_container_width=True):
            if not all([username, email, password, password2]):
                st.error("Please fill in all fields.")
            elif password != password2:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                ok, msg, _ = register_user(username, email, password, insta)
                if ok:
                    st.success(f"✅ {msg} Please sign in.")
                    st.session_state.auth_mode = "login"
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:#5858A8;font-size:13px;">Already have an account?</div>', unsafe_allow_html=True)
        if st.button("← Back to Sign In", use_container_width=True, key="goto_login"):
            st.session_state.auth_mode = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


if not st.session_state.logged_in:
    if st.session_state.auth_mode == "login":
        show_login()
    else:
        show_register()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
user    = st.session_state.user
user_id = user["id"]
handle  = user.get("instagram_handle") or f"@{user['username']}"
is_demo = user.get("is_demo", 0)

with st.sidebar:
    st.markdown(f"""
    <div class="logo-area">
        <div class="logo-text">📊 InstaSense</div>
        <div class="logo-tagline">AI Instagram Analytics</div>
    </div>
    <div style="padding:12px 16px;margin-bottom:16px;background:rgba(139,92,246,0.08);
                border-radius:10px;border:1px solid #2A2A40;">
        <div style="font-size:13px;font-weight:600;color:#C4B5FD;">{handle}</div>
        <div style="font-size:11px;color:#5858A8;margin-top:2px;">{user['email']}</div>
        {"<div style='margin-top:6px;'><span class='badge badge-demo'>Demo Account</span></div>" if is_demo else ""}
    </div>
    """, unsafe_allow_html=True)

    for label, key in [("🏠  Home","Home"),("➕  Add Post","AddPost"),("📁  My Posts","MyPosts"),("📈  Analytics","Analytics"),("💡  Insights","Insights")]:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    df_check = get_user_posts(user_id)
    if df_check.empty:
        st.warning("No posts yet.\nGo to **Add Post** to start!")
    else:
        st.success(f"✅ {len(df_check)} posts tracked")

    st.markdown("---")
    if st.button("🚪 Sign Out", use_container_width=True, key="signout"):
        for k in ["logged_in","user","caption_text"]:
            st.session_state[k] = False if k=="logged_in" else None if k=="user" else ""
        st.session_state.auth_mode = "login"
        st.rerun()

    st.markdown('<div style="font-size:11px;color:#3838A8;text-align:center;padding:8px;">InstaSense v3.0 • Emoji-Aware AI</div>', unsafe_allow_html=True)


@st.cache_data(ttl=30)
def load_posts_cached(uid):
    return get_user_posts(uid)

df_raw = load_posts_cached(user_id)
data_available = not df_raw.empty
if data_available and "engagement_score" not in df_raw.columns:
    df_raw["engagement_score"] = df_raw["likes"] + 2 * df_raw["comments"]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Home":
    st.markdown(f"""<div class="page-header">
        <h1 class="page-title">Dashboard Overview</h1>
        <p class="page-subtitle">Your personal analytics for <b style="color:#C4B5FD;">{handle}</b></p>
    </div>""", unsafe_allow_html=True)

    if not data_available:
        st.markdown("""<div class="empty-state">
            <div class="empty-state-icon">📸</div>
            <div class="empty-state-title">No posts yet</div>
            <div class="empty-state-sub">Add your first Instagram post to see analytics</div>
        </div>""", unsafe_allow_html=True)
        if st.button("➕ Add Your First Post"):
            st.session_state.page = "AddPost"; st.rerun()
    else:
        kpis = compute_kpis(df_raw); cat_df = compute_category_summary(df_raw)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="kpi-card" style="--accent:linear-gradient(90deg,#8B5CF6,#6366F1);">
                <span class="kpi-icon">📋</span><div class="kpi-value">{kpis['total_posts']}</div>
                <div class="kpi-label">Total Posts</div><div class="kpi-sub">{kpis['total_likes']:,} total likes</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="kpi-card" style="--accent:linear-gradient(90deg,#EC4899,#F43F5E);">
                <span class="kpi-icon">⚡</span><div class="kpi-value">{kpis['avg_engagement']:,.0f}</div>
                <div class="kpi-label">Avg Engagement</div><div class="kpi-sub">Likes + 2× Comments</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="kpi-card" style="--accent:linear-gradient(90deg,#F59E0B,#F97316);">
                <span class="kpi-icon">🏆</span><div class="kpi-value">{kpis['top_category']}</div>
                <div class="kpi-label">Top Category</div><div class="kpi-sub">Highest avg engagement</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="kpi-card" style="--accent:linear-gradient(90deg,#10B981,#14B8A6);">
                <span class="kpi-icon">😊</span><div class="kpi-value">{kpis['avg_sentiment']}%</div>
                <div class="kpi-label">Avg Sentiment</div><div class="kpi-sub">{kpis['positive_rate']}% positive posts</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if not cat_df.empty:
            col_l, col_r = st.columns([3, 2])
            with col_l:
                st.markdown('<div class="section-title">Engagement by Category</div>', unsafe_allow_html=True)
                fig = go.Figure(go.Bar(
                    x=cat_df["category"], y=cat_df["avg_engagement"],
                    marker=dict(color=[CATEGORY_COLORS.get(c,"#8B5CF6") for c in cat_df["category"]], opacity=0.9),
                    text=cat_df["avg_engagement"].apply(lambda v: f"{v:,.0f}"),
                    textposition="outside", textfont=dict(color="#9898C8", size=11),
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with col_r:
                st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
                sc = df_raw["sentiment_label"].value_counts()
                pos_pct = round(sc.get("Positive",0)/len(df_raw)*100,1)
                fig_p = go.Figure(go.Pie(labels=sc.index.tolist(), values=sc.values.tolist(), hole=0.62,
                    marker=dict(colors=["#8B5CF6","#EF5350","#888888"], line=dict(color="#0A0A0F",width=3))))
                fig_p.add_annotation(text=f"<b>{pos_pct}%</b><br><span style='font-size:10px'>Positive</span>",
                    x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="#E8E8F0", family="DM Sans"))
                fig_p.update_layout(**PLOTLY_LAYOUT, height=300)
                st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

        if len(df_raw) >= 3:
            st.markdown('<div class="section-title">Posts Over Time</div>', unsafe_allow_html=True)
            dt = df_raw.copy()
            dt["post_date"] = pd.to_datetime(dt["post_date"], errors="coerce")
            ts = dt.dropna(subset=["post_date"]).set_index("post_date").resample("W").agg(
                post_count=("id","count"), avg_eng=("engagement_score","mean")).reset_index()
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(x=ts["post_date"], y=ts["post_count"], name="Posts",
                mode="lines+markers", line=dict(color="#8B5CF6", width=2),
                fill="tozeroy", fillcolor="rgba(139,92,246,0.08)"))
            fig_l.update_layout(**PLOTLY_LAYOUT, height=220)
            st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ADD POST — with Emoji Pick
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "AddPost":
    st.markdown("""<div class="page-header">
        <h1 class="page-title">Add Instagram Post</h1>
        <p class="page-subtitle">Write your caption with the emoji picker — AI analyzes sentiment automatically</p>
    </div>""", unsafe_allow_html=True)

    col_form, col_emoji = st.columns([3, 2])

    with col_form:
        with st.form("add_post_form", clear_on_submit=True):
            caption = st.text_area(
                "Caption ✍️",
                value=st.session_state.caption_text,
                placeholder="Write your Instagram caption here... use the emoji picker on the right! →",
                height=150,
                key="caption_input",
            )
            category  = st.selectbox("Category", ["Fashion","Travel","Food","Fitness","Tech","Lifestyle"])
            col_l2, col_r2 = st.columns(2)
            with col_l2:
                likes     = st.number_input("Likes",     min_value=0, max_value=10_000_000, value=0, step=1)
                post_date = st.date_input("Post Date", value=datetime.date.today())
            with col_r2:
                comments  = st.number_input("Comments", min_value=0, max_value=1_000_000,  value=0, step=1)

            eng_preview = int(likes) + 2 * int(comments)
            st.markdown(f"""<div style="background:#0F0F1A;border:1px solid #1E1E30;border-radius:10px;
                        padding:12px 16px;margin:8px 0;">
                <span style="color:#6868A8;font-size:12px;text-transform:uppercase;">Engagement Preview</span>
                <span style="color:#C4B5FD;font-weight:700;font-size:18px;margin-left:12px;">⚡ {eng_preview:,}</span>
            </div>""", unsafe_allow_html=True)

            submitted = st.form_submit_button("✨ Analyze & Save Post", use_container_width=True)
            if submitted:
                cap_val = caption or st.session_state.caption_text
                if not cap_val.strip():
                    st.error("Please enter a caption.")
                else:
                    with st.spinner("🤖 Analyzing sentiment (including emojis)..."):
                        sentiment = add_post(user_id=user_id, caption=cap_val, category=category,
                            likes=int(likes), comments=int(comments), post_date=str(post_date))
                        load_posts_cached.clear()
                        st.session_state.caption_text = ""

                    color = "#4CAF82" if sentiment=="Positive" else ("#EF5350" if sentiment=="Negative" else "#888888")
                    icon  = "😊" if sentiment=="Positive" else ("😟" if sentiment=="Negative" else "😐")
                    st.success(f"✅ Post saved! Sentiment detected: {icon} **{sentiment}**")

    with col_emoji:
        st.markdown("""<div style="background:#0F0F1A;border:1px solid #1E1E30;border-radius:14px;
                    padding:16px;height:100%;">
            <div style="color:#8B5CF6;font-size:12px;font-weight:700;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:12px;">😊 Emoji Picker</div>
            <div style="font-size:12px;color:#4848A8;margin-bottom:10px;">
                Click an emoji → it copies to clipboard. Paste into caption with Ctrl+V
            </div>
        """, unsafe_allow_html=True)

        # Tab selector for emoji categories
        emoji_tab_names = list(EMOJI_CATEGORIES.keys())
        selected_tab = st.selectbox("Category", emoji_tab_names, key="emoji_cat_select",
            index=emoji_tab_names.index(st.session_state.emoji_tab) if st.session_state.emoji_tab in emoji_tab_names else 0,
            label_visibility="collapsed")
        st.session_state.emoji_tab = selected_tab

        # Display emojis as clickable buttons using JS clipboard copy
        emojis = EMOJI_CATEGORIES[selected_tab]
        emoji_html = '<div style="display:flex;flex-wrap:wrap;gap:4px;padding:8px 0;max-height:260px;overflow-y:auto;">'
        for emoji in emojis:
            emoji_html += f'''<span onclick="navigator.clipboard.writeText('{emoji}').then(()=>{{
                this.style.background='rgba(139,92,246,0.3)';
                setTimeout(()=>this.style.background='transparent',400)
            }})" title="Copy {emoji}" style="font-size:24px;cursor:pointer;padding:4px 5px;
            border-radius:6px;transition:background 0.15s;display:inline-block;"
            onmouseover="this.style.background='rgba(139,92,246,0.15)'"
            onmouseout="this.style.background='transparent'">{emoji}</span>'''
        emoji_html += '</div>'

        st.markdown(emoji_html, unsafe_allow_html=True)
        st.markdown("""<div style="font-size:11px;color:#4848A8;margin-top:8px;padding:8px;
            background:rgba(139,92,246,0.05);border-radius:8px;">
            💡 <b>Tip:</b> Emojis like 🔥❤️😍✨ boost Positive sentiment.
            Emojis like 😤😡😞 signal Negative sentiment. The AI reads both text AND emojis!
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MY POSTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "MyPosts":
    st.markdown(f"""<div class="page-header">
        <h1 class="page-title">My Posts</h1>
        <p class="page-subtitle">All posts for <b style="color:#C4B5FD;">{handle}</b> — {len(df_raw)} total</p>
    </div>""", unsafe_allow_html=True)

    if not data_available:
        st.markdown("""<div class="empty-state">
            <div class="empty-state-icon">📭</div><div class="empty-state-title">No posts yet</div>
            <div class="empty-state-sub">Use "Add Post" to enter your Instagram data</div>
        </div>""", unsafe_allow_html=True)
    else:
        f1, f2, f3 = st.columns([2,2,3])
        with f1: sel_cat  = st.selectbox("Category", ["All"]+sorted(df_raw["category"].unique().tolist()))
        with f2: sel_sent = st.selectbox("Sentiment", ["All"]+sorted(df_raw["sentiment_label"].dropna().unique().tolist()))
        with f3: search   = st.text_input("🔍 Search captions", placeholder="keyword...")

        df_f = df_raw.copy()
        if sel_cat  != "All": df_f = df_f[df_f["category"]==sel_cat]
        if sel_sent != "All": df_f = df_f[df_f["sentiment_label"]==sel_sent]
        if search:            df_f = df_f[df_f["caption"].str.contains(search, case=False, na=False)]

        st.markdown(f"<div style='color:#5858A8;font-size:13px;margin-bottom:16px;'>Showing {len(df_f)} of {len(df_raw)} posts</div>", unsafe_allow_html=True)

        for _, row in df_f.iterrows():
            cc   = CATEGORY_COLORS.get(row.get("category",""), "#8B5CF6")
            bcls = "badge-positive" if row.get("sentiment_label")=="Positive" else "badge-negative"
            cap  = str(row.get("caption",""))[:150] + ("…" if len(str(row.get("caption","")))>150 else "")
            cm, cd = st.columns([11,1])
            with cm:
                st.markdown(f"""<div style="background:#0F0F1A;border:1px solid #1E1E30;border-left:4px solid {cc};
                        border-radius:12px;padding:16px 20px;margin:6px 0;">
                    <div style="display:flex;justify-content:space-between;gap:12px;">
                        <div style="flex:1;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
                                <span style="color:{cc};font-size:12px;font-weight:600;background:rgba(139,92,246,0.1);padding:2px 10px;border-radius:20px;">{row.get('category','')}</span>
                                <span class="badge {bcls}">{row.get('sentiment_label','')}</span>
                                <span style="color:#4848A8;font-size:12px;">{row.get('post_date','')}</span>
                            </div>
                            <div style="color:#B8B8D8;font-size:13px;line-height:1.6;">{cap}</div>
                        </div>
                        <div style="text-align:right;min-width:120px;flex-shrink:0;">
                            <div style="font-size:12px;color:#6868A8;">❤️ {row.get('likes',0):,} &nbsp; 💬 {row.get('comments',0):,}</div>
                            <div style="font-size:14px;color:#C4B5FD;font-weight:700;margin-top:4px;">⚡ {row.get('engagement_score',0):,.0f}</div>
                            <div style="font-size:11px;color:#4848A8;margin-top:2px;">AI: {row.get('sentiment_score',0):.2f}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
            with cd:
                st.markdown("<div style='padding-top:20px;'>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_{row['id']}", help="Delete"):
                    delete_post(int(row["id"]), user_id); load_posts_cached.clear(); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        csv_b = df_f[["caption","category","likes","comments","engagement_score","sentiment_label","sentiment_score","post_date"]].to_csv(index=False).encode()
        st.download_button("⬇️ Download My Data as CSV", data=csv_b, file_name=f"instasense_{user['username']}.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Analytics":
    st.markdown(f"""<div class="page-header">
        <h1 class="page-title">Analytics</h1>
        <p class="page-subtitle">Deep dive into content performance for <b style="color:#C4B5FD;">{handle}</b></p>
    </div>""", unsafe_allow_html=True)

    if not data_available:
        st.markdown("""<div class="empty-state"><div class="empty-state-icon">📊</div>
            <div class="empty-state-title">No data yet</div>
            <div class="empty-state-sub">Add at least 3 posts to unlock analytics</div>
        </div>""", unsafe_allow_html=True)
    else:
        cat_df = compute_category_summary(df_raw)
        if not cat_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-title">Avg Likes by Category</div>', unsafe_allow_html=True)
                fig = px.bar(cat_df.sort_values("avg_likes"), x="avg_likes", y="category",
                    orientation="h", color="category", color_discrete_map=CATEGORY_COLORS, text="avg_likes")
                fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
                fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            with c2:
                st.markdown('<div class="section-title">Sentiment Score by Category</div>', unsafe_allow_html=True)
                sc2 = cat_df.sort_values("avg_sentiment")
                fig2 = px.bar(sc2, x="avg_sentiment", y="category", orientation="h",
                    color="avg_sentiment", color_continuous_scale=["#EF5350","#8B5CF6","#10B981"],
                    text=sc2["avg_sentiment"].apply(lambda v: f"{v*100:.1f}%"))
                fig2.update_traces(textposition="outside")
                fig2.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

            if len(cat_df) >= 2:
                st.markdown('<div class="section-title">Engagement vs Sentiment</div>', unsafe_allow_html=True)
                fig_s = px.scatter(cat_df, x="avg_sentiment", y="avg_engagement", size="total_posts",
                    color="category", color_discrete_map=CATEGORY_COLORS, text="category",
                    hover_data=["avg_likes","avg_comments","positive_rate"])
                fig_s.update_traces(textposition="top center", marker=dict(opacity=0.85))
                fig_s.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False)
                st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})

            st.markdown('<div class="section-title">Category Rankings</div>', unsafe_allow_html=True)
            d = cat_df[["rank","category","total_posts","avg_likes","avg_comments","avg_engagement","avg_sentiment","positive_rate"]].copy()
            d.columns = ["Rank","Category","Posts","Avg Likes","Avg Comments","Avg Engagement","Sentiment","Positive %"]
            d["Sentiment"]  = (d["Sentiment"]*100).round(1).astype(str)+"%"
            d["Positive %"] = d["Positive %"].astype(str)+"%"
            st.dataframe(d, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Top 5 Posts by Engagement</div>', unsafe_allow_html=True)
        for _, row in df_raw.nlargest(5, "engagement_score").iterrows():
            cc   = CATEGORY_COLORS.get(row.get("category",""), "#8B5CF6")
            bcls = "badge-positive" if row.get("sentiment_label")=="Positive" else "badge-negative"
            cap  = str(row.get("caption",""))[:120]
            st.markdown(f"""<div style="background:#0F0F1A;border:1px solid #1E1E30;border-left:4px solid {cc};
                    border-radius:12px;padding:16px 20px;margin:8px 0;">
                <div style="display:flex;justify-content:space-between;gap:16px;">
                    <div style="flex:1;">
                        <div style="font-size:12px;color:{cc};margin-bottom:6px;font-weight:600;">
                            {row.get('category','')} &nbsp;•&nbsp; <span style="color:#4848A8;">{row.get('post_date','')}</span>
                        </div>
                        <div style="color:#B8B8D8;font-size:13px;line-height:1.5;">{cap}</div>
                    </div>
                    <div style="text-align:right;min-width:130px;flex-shrink:0;">
                        <span class="badge {bcls}">{row.get('sentiment_label','')}</span>
                        <div style="font-size:12px;color:#6868A8;margin-top:8px;">❤️ {row.get('likes',0):,} &nbsp; 💬 {row.get('comments',0):,}</div>
                        <div style="font-size:14px;color:#C4B5FD;font-weight:700;">⚡ {row.get('engagement_score',0):,.0f}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Insights":
    st.markdown(f"""<div class="page-header">
        <h1 class="page-title">AI Insights</h1>
        <p class="page-subtitle">Personalized recommendations for <b style="color:#C4B5FD;">{handle}</b></p>
    </div>""", unsafe_allow_html=True)

    if not data_available:
        st.markdown("""<div class="empty-state"><div class="empty-state-icon">💡</div>
            <div class="empty-state-title">No insights yet</div>
            <div class="empty-state-sub">Add your posts — insights come from YOUR real data only</div>
        </div>""", unsafe_allow_html=True)
    else:
        kpis   = compute_kpis(df_raw)
        cat_df = compute_category_summary(df_raw)
        recs   = generate_recs(cat_df, len(df_raw), handle)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="kpi-card" style="--accent:linear-gradient(90deg,#8B5CF6,#6366F1);">
                <span class="kpi-icon">🏆</span><div class="kpi-value">{kpis['top_category']}</div>
                <div class="kpi-label">Top Category</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="kpi-card" style="--accent:linear-gradient(90deg,#10B981,#14B8A6);">
                <span class="kpi-icon">😊</span><div class="kpi-value">{kpis['positive_rate']}%</div>
                <div class="kpi-label">Positive Rate</div></div>""", unsafe_allow_html=True)
        with c3:
            bot = cat_df.iloc[-1]["category"] if not cat_df.empty else "—"
            st.markdown(f"""<div class="kpi-card" style="--accent:linear-gradient(90deg,#EF5350,#F43F5E);">
                <span class="kpi-icon">📉</span><div class="kpi-value">{bot}</div>
                <div class="kpi-label">Needs Attention</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Personalized Recommendations</div>', unsafe_allow_html=True)
        st.markdown(f"""<div style="background:#0A0A18;border:1px solid #1E1E30;border-radius:10px;
                padding:12px 18px;margin-bottom:16px;font-size:13px;color:#5858A8;">
            ℹ️ Based on <b style="color:#8888B8;">{len(df_raw)} posts</b> for
            <b style="color:#C4B5FD;">{handle}</b> — add more posts for more accurate insights.
        </div>""", unsafe_allow_html=True)

        for rec in recs:
            rh = rec
            while "**" in rh: rh = rh.replace("**","<b>",1).replace("**","</b>",1)
            st.markdown(f'<div class="rec-card">{rh}</div>', unsafe_allow_html=True)

        if "sentiment_label" in df_raw.columns and len(df_raw) >= 4:
            st.markdown('<div class="section-title">Engagement Heatmap (Category × Sentiment)</div>', unsafe_allow_html=True)
            pivot = df_raw.groupby(["category","sentiment_label"])["engagement_score"].mean().unstack(fill_value=0)
            fig_h = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale=[[0,"#0F0F1A"],[0.5,"#4C1D95"],[1,"#8B5CF6"]],
                text=np.round(pivot.values,0), texttemplate="%{text:,.0f}"))
            fig_h.update_layout(**PLOTLY_LAYOUT, height=280)
            st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar":False})

        st.markdown('<div class="section-title">Content Strategy Matrix</div>', unsafe_allow_html=True)
        st.markdown("""<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div style="background:#0F0F1A;border:1px solid #1E1E30;border-radius:12px;padding:20px;">
                <div style="color:#8B5CF6;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">🚀 Double Down</div>
                <div style="color:#8888B8;font-size:13px;line-height:1.7;">High engagement + high sentiment. Post more, invest in quality visuals and production.</div>
            </div>
            <div style="background:#0F0F1A;border:1px solid #1E1E30;border-radius:12px;padding:20px;">
                <div style="color:#EC4899;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">💡 Optimize</div>
                <div style="color:#8888B8;font-size:13px;line-height:1.7;">High engagement, mixed sentiment. Improve caption tone and respond to your community.</div>
            </div>
            <div style="background:#0F0F1A;border:1px solid #1E1E30;border-radius:12px;padding:20px;">
                <div style="color:#14B8A6;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">🌱 Grow</div>
                <div style="color:#8888B8;font-size:13px;line-height:1.7;">Great sentiment, lower reach. Use targeted hashtags, collaborations, and paid boosts.</div>
            </div>
            <div style="background:#0F0F1A;border:1px solid #1E1E30;border-radius:12px;padding:20px;">
                <div style="color:#F59E0B;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">🔄 Reconsider</div>
                <div style="color:#8888B8;font-size:13px;line-height:1.7;">Low engagement + low sentiment. Rethink the entire content angle, format, or audience.</div>
            </div>
        </div>""", unsafe_allow_html=True)