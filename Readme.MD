# 📊 InstaSense — AI Instagram Analytics Dashboard

A production-level Instagram analytics platform powered by DistilBERT sentiment analysis, SQLite, and Streamlit.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (generates data + runs AI analysis)
```bash
python run_pipeline.py
```
This will:
- Generate 7,500 synthetic Instagram posts
- Run DistilBERT sentiment analysis on all captions
- Store results in SQLite
- Compute analytics and print a summary

> **Note:** First run downloads the DistilBERT model (~268MB). Subsequent runs use the HuggingFace cache.

### 3. Launch the dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
instasense/
├── app.py                  # Streamlit dashboard (4 pages)
├── run_pipeline.py         # Full pipeline orchestrator
├── generate_dataset.py     # Synthetic dataset generator
├── requirements.txt        # Python dependencies
├── data/
│   ├── instagram_posts.csv # Generated dataset
│   └── instasense.db       # SQLite database
└── modules/
    ├── sentiment.py        # DistilBERT sentiment analysis
    ├── database.py         # SQLite CRUD operations
    └── analytics.py        # Metrics, rankings, recommendations
```

---

## 🧠 AI Model

**distilbert-base-uncased-finetuned-sst-2-english**
- 67M parameter transformer fine-tuned for sentiment classification
- Returns: `Positive` / `Negative` label + confidence score
- Text preprocessing: emoji normalization, hashtag cleaning, truncation

---

## 📈 Dashboard Pages

| Page | Description |
|------|-------------|
| **Home** | KPI cards, engagement bar chart, sentiment pie, time series |
| **Dataset** | Browse, filter, search, and download posts |
| **Analytics** | Category rankings, scatter plots, top 5 posts |
| **Insights** | AI recommendations, heatmap, content strategy matrix |

---

## ⚙️ Pipeline Options

```bash
# Generate 10,000 records instead of 7,500
python run_pipeline.py --records 10000

# Use existing CSV (skip dataset generation)
python run_pipeline.py --skip-generate

# Adjust inference batch size (larger = faster on GPU)
python run_pipeline.py --batch-size 128
```

---

## 🎨 Tech Stack

- **ML**: HuggingFace Transformers (DistilBERT)
- **Database**: SQLite via Python stdlib
- **Analytics**: Pandas + NumPy
- **Visualizations**: Plotly
- **Frontend**: Streamlit with custom CSS (dark theme)
- **Fonts**: DM Sans + DM Mono (Google Fonts)