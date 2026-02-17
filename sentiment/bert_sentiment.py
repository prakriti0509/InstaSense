"""
InstaSense - Advanced Sentiment Analysis Module
Uses distilbert-base-uncased-finetuned-sst-2-english for fast,
accurate sentiment analysis with robust text preprocessing.
"""

import re
import logging
from typing import Optional
import pandas as pd
from transformers import pipeline, Pipeline

logger = logging.getLogger(__name__)

# ── Emoji-to-sentiment mapping for preprocessing ─────────────────────────────
POSITIVE_EMOJIS = {
    "😍", "🔥", "💯", "✨", "🌟", "💫", "🎉", "🙌", "💪", "❤️",
    "😊", "🥰", "😁", "🎊", "👏", "💚", "💜", "💙", "🧡", "💛",
    "🌅", "🌺", "🏅", "🥗", "🍝", "🤩", "🫶", "🚀",
}
NEGATIVE_EMOJIS = {
    "😤", "😡", "😞", "😔", "😢", "💀", "👎", "😠", "🙄", "😒",
}


def clean_text(text: str) -> str:
    """
    Clean and normalize Instagram caption text for sentiment analysis.

    Steps:
      1. Strip leading/trailing whitespace
      2. Replace emoji sentiment hints with descriptive words
      3. Remove non-ASCII characters (remaining emojis, special symbols)
      4. Remove hashtags and mentions (preserve the word content where useful)
      5. Remove URLs
      6. Collapse multiple spaces
      7. Truncate to 512 characters (model token limit)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert positive/negative emojis to placeholder words
    for emoji in POSITIVE_EMOJIS:
        text = text.replace(emoji, " great ")
    for emoji in NEGATIVE_EMOJIS:
        text = text.replace(emoji, " terrible ")

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove hashtags (keep the word) — e.g., #FashionForward → FashionForward
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove @mentions entirely
    text = re.sub(r"@\w+", "", text)

    # Remove non-ASCII characters (remaining emojis, symbols)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Collapse multiple whitespace/newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate to 512 characters
    return text[:512]


def load_sentiment_model() -> Pipeline:
    """
    Load the DistilBERT SST-2 sentiment pipeline.
    Downloads model on first run and caches locally via HuggingFace.
    Falls back to VADER if the model cannot be downloaded.
    """
    logger.info("Loading DistilBERT sentiment model...")
    print("🤖 Loading sentiment model (distilbert-base-uncased-finetuned-sst-2-english)...")
    print("   (First run downloads ~268MB model — subsequent runs use cache)")

    try:
        sentiment_pipeline = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        print("✅ Model loaded successfully.\n")
        return sentiment_pipeline
    except Exception as e:
        logger.warning(f"DistilBERT unavailable ({e}). Falling back to VADER.")
        print("⚠️  DistilBERT unavailable — using VADER fallback (no internet required).\n")
        return _load_vader_fallback()


def _load_vader_fallback():
    """
    VADER-based fallback when DistilBERT cannot be downloaded.
    Returns a callable with the same interface as a HuggingFace pipeline.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    # Positive/negative keyword boosters that survive emoji stripping
    POS_WORDS = {
        "great", "obsessed", "amazing", "love", "best", "perfect", "incredible",
        "outstanding", "wonderful", "fantastic", "awesome", "beautiful", "proud",
        "milestone", "record", "win", "excited", "magical", "heals", "transformed",
        "achieved", "complete", "worth", "joy", "grateful", "delicious", "stunning",
    }
    NEG_WORDS = {
        "terrible", "worst", "never", "hate", "disappointed", "frustrating", "broken",
        "overpriced", "cold", "disaster", "wasted", "scam", "ruined", "ugly", "failed",
        "terrible", "toxic", "burnout", "injured", "setback", "overrated", "miserable",
    }

    class VaderPipeline:
        def __call__(self, texts):
            results = []
            for text in texts:
                scores = analyzer.polarity_scores(text)
                compound = scores["compound"]

                # Boost compound with keyword heuristics
                words = set(text.lower().split())
                pos_boost = len(words & POS_WORDS) * 0.08
                neg_boost = len(words & NEG_WORDS) * 0.08
                compound = max(-1.0, min(1.0, compound + pos_boost - neg_boost))

                if compound >= 0.03:
                    results.append({"label": "POSITIVE", "score": round(min(0.5 + compound / 2, 0.99), 4)})
                else:
                    results.append({"label": "NEGATIVE", "score": round(min(0.5 + abs(compound) / 2, 0.99), 4)})
            return results

    return VaderPipeline()


def analyze_batch(
    texts: list[str],
    model: Pipeline,
    batch_size: int = 64,
) -> list[dict]:
    """
    Run sentiment analysis on a batch of texts.

    Args:
        texts: List of raw caption strings
        model: Loaded HuggingFace pipeline
        batch_size: Number of texts per inference batch

    Returns:
        List of dicts with keys: sentiment_label, sentiment_score
    """
    results = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch_raw = texts[i : i + batch_size]
        batch_clean = [clean_text(t) for t in batch_raw]

        # Replace empty strings with neutral placeholder
        batch_for_model = [t if t else "no content" for t in batch_clean]

        try:
            batch_results = model(batch_for_model)
            for raw, clean_t, res in zip(batch_raw, batch_clean, batch_results):
                results.append({
                    "sentiment_label": res["label"].capitalize(),   # POSITIVE → Positive
                    "sentiment_score": round(res["score"], 4),
                })
        except Exception as e:
            logger.warning(f"Batch {i//batch_size} failed: {e}. Filling with neutral.")
            for _ in batch_raw:
                results.append({"sentiment_label": "Neutral", "sentiment_score": 0.5})

        # Progress logging every 10 batches
        processed = min(i + batch_size, total)
        if (i // batch_size) % 10 == 0:
            pct = processed / total * 100
            print(f"   🔍 Sentiment analysis: {processed}/{total} ({pct:.1f}%)")

    return results


def run_sentiment_analysis(
    df: pd.DataFrame,
    text_column: str = "caption",
    batch_size: int = 64,
    model: Optional[Pipeline] = None,
) -> pd.DataFrame:
    """
    Add sentiment columns to a DataFrame.

    Args:
        df: DataFrame containing Instagram posts
        text_column: Column name holding caption text
        batch_size: Inference batch size
        model: Pre-loaded pipeline (loads one if None)

    Returns:
        DataFrame with added sentiment_label and sentiment_score columns
    """
    if model is None:
        model = load_sentiment_model()

    texts = df[text_column].fillna("").tolist()
    print(f"🔄 Analyzing sentiment for {len(texts)} posts...")

    results = analyze_batch(texts, model, batch_size=batch_size)
    results_df = pd.DataFrame(results)

    df = df.copy()
    df["sentiment_label"] = results_df["sentiment_label"].values
    df["sentiment_score"] = results_df["sentiment_score"].values

    pos = (df["sentiment_label"] == "Positive").sum()
    neg = (df["sentiment_label"] == "Negative").sum()
    print(f"\n✅ Sentiment analysis complete.")
    print(f"   Positive: {pos} ({pos/len(df)*100:.1f}%) | Negative: {neg} ({neg/len(df)*100:.1f}%)")

    return df


if __name__ == "__main__":
    # Quick smoke test
    sample_texts = [
        "This is the most amazing view I've ever seen! 🌅 #Travel",
        "Terrible experience. Never going back. 😡",
        "",
        "Just another day. Nothing special happened.",
        "🔥🔥🔥 Obsessed with this look! #Fashion",
    ]

    model = load_sentiment_model()
    results = analyze_batch(sample_texts, model)
    for text, result in zip(sample_texts, results):
        print(f"  Text: {text[:60]!r:60s} → {result}")