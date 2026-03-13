import pandas as pd

def process_news():

    df = pd.read_csv("data/raw/news_raw.csv")

    df["date"] = pd.to_datetime(df["date"])

    daily_count = (
        df.groupby("date")
          .size()
          .reset_index(name="news_count")
    )

    daily_count.to_csv("data/processed/news_daily_count.csv", index=False)

    return daily_count
