import os
import re
import string
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# Load The API Keys 
load_dotenv()
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_KEY")

if not NEWS_API_KEY:
    raise RuntimeError("NEWSAPI_KEY not found. Please add it to .env file.")

#Text Cleaning
def clean_text(text: str) -> str:
    """Remove URLs, HTML tags, punctuation, and extra spaces for cleaner analysis."""
    text = text or ""
    text = re.sub(r"http\S+|<.*?>|\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

# Fetch The News
def fetch_articles(query="artificial intelligence", days=7, target=100) -> pd.DataFrame:
    """Fetch articles from NewsAPI + GNews, remove duplicates, return as DataFrame."""
    since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    params = {
        "q": query,
        "from": since,
        "language": "en",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY
    }
    print(" Fetching from NewsAPI...")
    news_resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10).json()
    articles = news_resp.get("articles", [])

    if len(articles) < target and GNEWS_API_KEY:
        left = target - len(articles)
        print(f" Fetching extra {left} from GNews...")
        g_resp = requests.get(
            "https://gnews.io/api/v4/search",
            params={"q": query, "lang": "en", "max": left, "token": GNEWS_API_KEY},
            timeout=10
        ).json()
        articles += g_resp.get("articles", [])

    seen_urls, records = set(), []
    for art in articles:
        url = art.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        records.append({
            "source": art.get("source", {}).get("name", "Unknown"),
            "title": (art.get("title") or "").strip(),
            "description": (art.get("description") or "").strip(),
            "url": url,
            "published_at": art.get("publishedAt", "")
        })
    return pd.DataFrame(records)

#Sentiment Analysis 
def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Apply VADER to get sentiment score & label for each article."""
    analyzer = SentimentIntensityAnalyzer()
    scores, labels = [], []

    for _, row in df.iterrows():
        text = clean_text(row["title"] + " " + row["description"])
        score = analyzer.polarity_scores(text)["compound"]
        scores.append(score)
        if score >= 0.05:
            labels.append("Positive")
        elif score <= -0.05:
            labels.append("Negative")
        else:
            labels.append("Neutral")

    df["score"] = scores
    df["sentiment"] = labels
    return df

#Visualizations
def save_results_and_visuals(df: pd.DataFrame):
    """Save to CSV and generate 5 different insightful plots."""
    df.to_csv("ai_news_sentiment.csv", index=False)
    print(f" Saved {len(df)} rows to ai_news_sentiment.csv")

    #The Graph or plot of Sentiment counts
    order = ["Positive", "Neutral", "Negative"]
    counts = df["sentiment"].value_counts().reindex(order).fillna(0)
    counts.plot.bar(color=["green", "gray", "red"])
    plt.title("Sentiment Counts")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.savefig("plot1_sentiment_counts.png")
    plt.clf()

    #The Graph or plot of Sentiment trend over time
    df["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.date
    trend = df.dropna(subset=["date"]).groupby("date")["score"].mean()
    trend.plot(marker="o", color="blue")
    plt.title("Average Sentiment Score Over Time")
    plt.ylabel("Compound Score")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("plot2_sentiment_trend.png")
    plt.clf()

    # Image of Word cloud of titles
    text = " ".join(df["title"].dropna())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Headlines")
    plt.tight_layout()
    plt.savefig("plot3_wordcloud.png")
    plt.clf()

    # Graph of Top 10 sources by number of articles
    top_sources = df["source"].value_counts().nlargest(10)
    top_sources.plot.barh(color="teal")
    plt.title("Top 10 News Sources by Article Count")
    plt.xlabel("Number of Articles")
    plt.tight_layout()
    plt.savefig("plot4_top_sources.png")
    plt.clf()

    # Plot of Sentiment distribution per source (stacked)
    sentiment_by_source = (
        df.groupby(["source", "sentiment"]).size().unstack(fill_value=0).head(10)
    )
    sentiment_by_source.plot(kind="bar", stacked=True, figsize=(10, 6),
                             color={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    plt.title("Sentiment Distribution by Top Sources")
    plt.ylabel("Article Count")
    plt.tight_layout()
    plt.savefig("plot5_sentiment_by_source.png")
    plt.clf()

    print("5 plots generated (plot1_... to plot5_...).")

# Main Function
def main():
    print(" Task 1: Fetching & Analyzing News")
    df = fetch_articles()
    print(f" Retrieved {len(df)} unique articles")
    df = analyze_sentiment(df)
    save_results_and_visuals(df)

if __name__ == "__main__":
    main()
