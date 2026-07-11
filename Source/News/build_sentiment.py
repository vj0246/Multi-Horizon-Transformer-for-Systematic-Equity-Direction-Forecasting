"""Build a daily FinBERT sentiment series for fusion into the Transformer.

Pipeline:  NewsAPI headlines  ->  FinBERT score per headline  ->  daily mean
->  Data/Processed_Data/daily_sentiment.csv  (columns: date, daily_sentiment)

Run:
    NEWSAPI_KEY=xxxx python -m Source.News.build_sentiment --query "Nifty 50 India stock market"

IMPORTANT — coverage limitation:
    NewsAPI's free/standard tiers only return roughly the last 30 days of articles.
    A real sentiment feature therefore CANNOT be backfilled across the 2007-2026
    training window. This script produces genuine sentiment for whatever window the
    API returns; it is wired into the model (features.use_sentiment) but kept OFF by
    default so the historical backtest never trains on fabricated/all-zero sentiment.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Data" / "Processed_Data" / "daily_sentiment.csv"


def _finbert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tok, model


def score_headlines(titles: list[str]) -> list[float]:
    """FinBERT scalar sentiment per headline: -P(negative) + P(positive), in [-1, 1]."""
    import torch

    tok, model = _finbert()
    scores = []
    for text in titles:
        inputs = tok(text or "", return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1).numpy()[0]
        scores.append(float(probs[0] * -1 + probs[2] * 1))  # [neg, neutral, pos]
    return scores


def fetch_headlines(query: str, api_key: str) -> pd.DataFrame:
    import requests

    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={"q": query, "language": "en", "sortBy": "publishedAt",
                "pageSize": 100, "apiKey": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return pd.DataFrame(
        [{"date": a["publishedAt"][:10], "title": a["title"]} for a in articles]
    )


def build(query: str, api_key: str) -> pd.DataFrame:
    df = fetch_headlines(query, api_key)
    if df.empty:
        raise SystemExit("NewsAPI returned no articles for that query.")
    df["sentiment"] = score_headlines(df["title"].fillna("").tolist())
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["sentiment"].mean().reset_index()
    daily.columns = ["date", "daily_sentiment"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT, index=False)
    print(f"Wrote {len(daily)} daily sentiment rows -> {OUT}")
    print(f"Coverage: {daily['date'].min().date()} .. {daily['date'].max().date()} "
          f"(NewsAPI history is limited; enable features.use_sentiment only if this "
          f"window overlaps your evaluation period).")
    return daily


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="Nifty 50 India stock market")
    args = ap.parse_args()
    key = os.environ.get("NEWSAPI_KEY")
    if not key:
        raise SystemExit("Set NEWSAPI_KEY in the environment first.")
    build(args.query, key)
