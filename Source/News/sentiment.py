from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np

def compute_sentiment():

    df = pd.read_csv("data/raw/news_raw.csv")

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    sentiments = []

    for text in df["title"].fillna(""):

        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]

        sentiment_score = probs[0] * -1 + probs[2] * 1  # negative + positive

        sentiments.append(sentiment_score)

    df["sentiment_score"] = sentiments
    df["date"] = pd.to_datetime(df["date"])

    daily_sentiment = (
        df.groupby("date")["sentiment_score"]
          .mean()
          .reset_index()
    )

    daily_sentiment.to_csv("data/processed/news_daily_sentiment.csv", index=False)

    return daily_sentiment
