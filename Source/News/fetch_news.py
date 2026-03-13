import requests
import pandas as pd
import os

def fetch_news(query, api_key):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    articles = []

    for article in data["articles"]:
        articles.append({
            "date": article["publishedAt"][:10],
            "title": article["title"],
            "description": article["description"]
        })

    df = pd.DataFrame(articles)

    os.makedirs("ddata/raw", exist_ok=True)
    df.to_csv("Data/Raw_Data/news_raw.csv", index=False)

    return df
