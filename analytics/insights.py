"""
InstaSense - Analytics Engine
Computes engagement metrics, rankings, sentiment distribution,
and generates actionable recommendations.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ── Core Metrics ──────────────────────────────────────────────────────────────

def compute_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engagement_score column.
    Formula: engagement_score = likes + (2 * comments)
    Comments are weighted more as they signal deeper user intent.
    """
    df = df.copy()
    df["engagement_score"] = df["likes"] + (2 * df["comments"])
    return df


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-category metrics.

    Returns DataFrame with columns:
        category, total_posts, avg_likes, avg_comments,
        avg_engagement_score, avg_sentiment_score,
        positive_rate, engagement_rank
    """
    df = compute_engagement_score(df)

    # Numeric sentiment score: Positive = score, Negative = 1 - score
    df["numeric_sentiment"] = df.apply(
        lambda r: r["sentiment_score"] if r["sentiment_label"] == "Positive" else 1 - r["sentiment_score"],
        axis=1,
    )

    summary = (
        df.groupby("category")
        .agg(
            total_posts=("post_id", "count"),
            avg_likes=("likes", "mean"),
            avg_comments=("comments", "mean"),
            avg_engagement_score=("engagement_score", "mean"),
            avg_sentiment_score=("numeric_sentiment", "mean"),
            positive_count=("sentiment_label", lambda x: (x == "Positive").sum()),
        )
        .reset_index()
    )

    summary["positive_rate"] = (summary["positive_count"] / summary["total_posts"] * 100).round(1)
    summary["avg_likes"] = summary["avg_likes"].round(0).astype(int)
    summary["avg_comments"] = summary["avg_comments"].round(0).astype(int)
    summary["avg_engagement_score"] = summary["avg_engagement_score"].round(1)
    summary["avg_sentiment_score"] = summary["avg_sentiment_score"].round(4)

    # Composite rank: 60% engagement + 40% sentiment
    eng_norm = (summary["avg_engagement_score"] - summary["avg_engagement_score"].min()) / (
        summary["avg_engagement_score"].max() - summary["avg_engagement_score"].min() + 1e-9
    )
    sent_norm = (summary["avg_sentiment_score"] - summary["avg_sentiment_score"].min()) / (
        summary["avg_sentiment_score"].max() - summary["avg_sentiment_score"].min() + 1e-9
    )
    summary["composite_score"] = (0.6 * eng_norm + 0.4 * sent_norm).round(4)
    summary["engagement_rank"] = summary["composite_score"].rank(ascending=False).astype(int)
    summary = summary.sort_values("engagement_rank")

    return summary.drop(columns=["positive_count"])


def top_posts(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the top N posts by engagement score."""
    df = compute_engagement_score(df)
    cols = ["post_id", "username", "category", "caption", "likes",
            "comments", "engagement_score", "sentiment_label", "sentiment_score", "post_date"]
    available = [c for c in cols if c in df.columns]
    return (
        df[available]
        .sort_values("engagement_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def sentiment_distribution(df: pd.DataFrame) -> dict:
    """Return counts and percentages for Positive / Negative sentiment."""
    counts = df["sentiment_label"].value_counts()
    total = len(df)
    return {
        label: {"count": int(counts.get(label, 0)), "pct": round(counts.get(label, 0) / total * 100, 1)}
        for label in ["Positive", "Negative"]
    }


def posts_over_time(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Aggregate post count and average engagement by time frequency.

    Args:
        df: Posts DataFrame
        freq: Pandas resample frequency ('D' daily, 'W' weekly, 'ME' monthly)
    """
    df = compute_engagement_score(df)
    df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
    df_valid = df.dropna(subset=["post_date"]).set_index("post_date")

    time_series = (
        df_valid.resample(freq)
        .agg(post_count=("post_id", "count"), avg_engagement=("engagement_score", "mean"))
        .reset_index()
    )
    time_series["avg_engagement"] = time_series["avg_engagement"].round(1)
    return time_series


# ── KPI Summary ───────────────────────────────────────────────────────────────

def kpi_summary(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPI cards for the dashboard.

    Returns dict with:
        total_posts, avg_engagement, avg_sentiment,
        top_category, positive_rate
    """
    df = compute_engagement_score(df)

    df["numeric_sentiment"] = df.apply(
        lambda r: r["sentiment_score"] if r["sentiment_label"] == "Positive" else 1 - r["sentiment_score"],
        axis=1,
    )

    cat_eng = df.groupby("category")["engagement_score"].mean()
    top_cat = cat_eng.idxmax()

    return {
        "total_posts": len(df),
        "avg_engagement": round(df["engagement_score"].mean(), 1),
        "avg_sentiment": round(df["numeric_sentiment"].mean() * 100, 1),
        "top_category": top_cat,
        "positive_rate": round((df["sentiment_label"] == "Positive").mean() * 100, 1),
    }


# ── Recommendation Engine ─────────────────────────────────────────────────────

def generate_recommendations(category_df: pd.DataFrame) -> list[str]:
    """
    Generate human-readable actionable recommendations.

    Args:
        category_df: Output of category_summary()

    Returns:
        List of recommendation strings
    """
    recs = []
    top = category_df.iloc[0]  # Rank 1 category
    bottom = category_df.iloc[-1]  # Lowest ranked category

    recs.append(
        f"🏆 Focus on **{top['category']}**: it leads with an avg engagement score of "
        f"{top['avg_engagement_score']:,.0f} and {top['positive_rate']}% positive sentiment."
    )

    # High sentiment but lower engagement
    high_sent = category_df[
        (category_df["avg_sentiment_score"] > 0.80) &
        (category_df["engagement_rank"] > 2)
    ]
    if not high_sent.empty:
        cat = high_sent.iloc[0]["category"]
        recs.append(
            f"💡 **{cat}** has excellent sentiment ({high_sent.iloc[0]['avg_sentiment_score']*100:.1f}% positive) "
            f"but moderate engagement — invest in higher-quality visuals to boost reach."
        )

    # High engagement, moderate sentiment — exclude the already-featured top category
    low_sent = category_df[
        (category_df["avg_sentiment_score"] < 0.70) &
        (category_df["engagement_rank"] <= 3) &
        (category_df["category"] != top["category"])
    ]
    if not low_sent.empty:
        cat = low_sent.iloc[0]["category"]
        recs.append(
            f"⚠️ **{cat}** drives strong engagement but lower sentiment scores — "
            f"focus on community response to improve brand perception."
        )

    recs.append(
        f"📉 **{bottom['category']}** ranks last — consider refreshing content strategy, "
        f"posting schedules, and hashtag targeting to improve discoverability."
    )

    # Posting cadence insight
    recs.append(
        "⏰ Post during peak hours (7–9am and 7–10pm local time) to maximize initial engagement velocity."
    )

    return recs


# ── Full Analytics Report ─────────────────────────────────────────────────────

def run_analytics(df: pd.DataFrame) -> dict:
    """
    Run the complete analytics pipeline.

    Returns:
        dict with keys: kpis, category_summary, top_posts,
                        sentiment_dist, time_series, recommendations
    """
    print("📊 Running analytics engine...")

    cat_summary = category_summary(df)
    kpis = kpi_summary(df)
    t5 = top_posts(df, n=5)
    sent_dist = sentiment_distribution(df)
    ts = posts_over_time(df, freq="W")
    recs = generate_recommendations(cat_summary)

    print(f"   ✅ KPIs computed | Top category: {kpis['top_category']}")
    print(f"   ✅ Category rankings computed")
    print(f"   ✅ {len(recs)} recommendations generated")

    return {
        "kpis": kpis,
        "category_summary": cat_summary,
        "top_posts": t5,
        "sentiment_dist": sent_dist,
        "time_series": ts,
        "recommendations": recs,
    }