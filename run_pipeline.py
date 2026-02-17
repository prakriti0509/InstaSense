"""
InstaSense - Full Pipeline Runner
Executes: generate_data → init_db → sentiment → insert → analytics
Run with: python run_pipeline.py
"""

import sys
import os
import time
import logging
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("instasense.pipeline")

# ── Color helpers ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def header(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}\n")


def success(text: str) -> None:
    print(f"{GREEN}✅ {text}{RESET}")


def warn(text: str) -> None:
    print(f"{YELLOW}⚠️  {text}{RESET}")


def error(text: str) -> None:
    print(f"{RED}❌ {text}{RESET}")


# ── Pipeline Steps ────────────────────────────────────────────────────────────

def step_generate_dataset(num_records: int = 7500) -> str:
    """Step 1: Generate synthetic dataset CSV."""
    header("STEP 1 — Generating Synthetic Dataset")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_data",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_data.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    generate_dataset = mod.generate_dataset

    csv_path = "data/instagram_posts.csv"
    if os.path.exists(csv_path):
        warn(f"Dataset already exists at {csv_path} — regenerating fresh copy.")

    df = generate_dataset(num_records=num_records, output_path=csv_path)
    success(f"Dataset generated: {len(df)} records → {csv_path}")
    return csv_path


def step_init_database() -> None:
    """Step 2: Initialize (or reset) the SQLite database."""
    header("STEP 2 — Initializing Database")
    from database import init_db
    init_db(reset=True)
    success("Database schema created with clean slate.")


def step_load_csv(csv_path: str):
    """Step 3: Load the generated CSV into a DataFrame."""
    header("STEP 3 — Loading CSV Data")
    import pandas as pd
    df = pd.read_csv(csv_path)
    success(f"Loaded {len(df)} rows from {csv_path}")
    return df


def step_sentiment_analysis(df, batch_size: int = 64):
    """Step 4: Run sentiment analysis on all captions."""
    header("STEP 4 — Sentiment Analysis (DistilBERT)")
    from sentiment import bert_sentiment
    df = bert_sentiment.run_sentiment_analysis(df, batch_size=batch_size)
    success("Sentiment analysis complete.")
    return df


def step_insert_to_db(df) -> None:
    """Step 5: Insert enriched posts into SQLite."""
    header("STEP 5 — Inserting Data into SQLite")
    from database import insert_posts
    inserted = insert_posts(df)
    success(f"Inserted {inserted} records into database.")


def step_run_analytics(df) -> dict:
    """Step 6: Run full analytics engine."""
    header("STEP 6 — Analytics Engine")
    from analytics import insights
    results = insights.run_analytics(df)

    kpis = results["kpis"]
    print(f"\n  📈 Total Posts:      {kpis['total_posts']:,}")
    print(f"  💥 Avg Engagement:   {kpis['avg_engagement']:,.1f}")
    print(f"  😊 Avg Sentiment:    {kpis['avg_sentiment']}%")
    print(f"  🏆 Top Category:     {kpis['top_category']}")
    print(f"  ✅ Positive Rate:    {kpis['positive_rate']}%")

    print(f"\n  📊 Category Rankings:")
    cat_df = results["category_summary"]
    for _, row in cat_df.iterrows():
        print(f"     #{int(row['engagement_rank'])} {row['category']:12s} — "
              f"Eng: {row['avg_engagement_score']:,.0f} | "
              f"Sentiment: {row['avg_sentiment_score']*100:.1f}%")

    print(f"\n  💡 Recommendations:")
    for rec in results["recommendations"]:
        clean = rec.replace("**", "")
        print(f"     {clean}")

    return results


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InstaSense Pipeline Runner")
    parser.add_argument("--records", type=int, default=7500, help="Number of records to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Sentiment inference batch size")
    parser.add_argument("--skip-generate", action="store_true", help="Skip dataset generation")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═' * 60}")
    print(f"  🚀 InstaSense — AI Instagram Analytics Pipeline")
    print(f"{'═' * 60}{RESET}\n")

    total_start = time.time()

    try:
        # Step 1: Generate data
        csv_path = "data/instagram_posts.csv"
        if not args.skip_generate or not os.path.exists(csv_path):
            csv_path = step_generate_dataset(num_records=args.records)
        else:
            warn(f"Skipping generation — using existing {csv_path}")

        # Step 2: Init DB
        step_init_database()

        # Step 3: Load CSV
        df = step_load_csv(csv_path)

        # Step 4: Sentiment
        df = step_sentiment_analysis(df, batch_size=args.batch_size)

        # Step 5: Insert to DB
        step_insert_to_db(df)

        # Step 6: Analytics
        step_run_analytics(df)

        elapsed = time.time() - total_start
        print(f"\n{BOLD}{GREEN}{'═' * 60}")
        print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
        print(f"  👉 Now run: streamlit run app.py")
        print(f"{'═' * 60}{RESET}\n")

    except Exception as e:
        error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()