import os
import re
import string
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv

#CONFIG
load_dotenv()
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

#TEXT CLEANING
def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"http\S+|<.*?>|\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

#FETCH LIVE ARTICLES 
def fetch_articles(query="artificial intelligence", days=7, limit=100):
    since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    params = {
        "q": query,
        "from": since,
        "language": "en",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY
    }
    print("Fetching live articles from NewsAPI...")
    resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10).json()
    articles = resp.get("articles", [])
    print(f"Retrieved {len(articles)} articles.")
    rows = []
    for art in articles:
        rows.append({
            "source": art.get("source", {}).get("name", "Unknown"),
            "title": art.get("title"),
            "description": art.get("description"),
            "url": art.get("url"),
            "published_at": art.get("publishedAt")
        })
    return pd.DataFrame(rows)

#SENTIMENT ANALYSIS
def analyze_sentiment(df):
    sid = SentimentIntensityAnalyzer()
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["score"] = df["text"].apply(lambda x: sid.polarity_scores(clean_text(x))["compound"])
    df["sentiment"] = np.where(df["score"] > 0.05, "Positive",
                        np.where(df["score"] < -0.05, "Negative", "Neutral"))
    df["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.date
    return df.dropna(subset=["date"])

#FORECASTING MODEL
def train_and_forecast(df, days_ahead=5):
    daily = df.groupby("date")["score"].mean().reset_index()
    daily["t"] = np.arange(len(daily))
    X, y = daily[["t"]], daily["score"]

    model = LinearRegression().fit(X, y)
    future_t = np.arange(len(daily), len(daily) + days_ahead).reshape(-1, 1)
    forecast = model.predict(future_t)
    future_dates = [daily["date"].iloc[-1] + timedelta(days=i+1) for i in range(days_ahead)]
    forecast_df = pd.DataFrame({"date": future_dates, "forecast_score": forecast})
    return daily, forecast_df

#  ALERT LOGIC
from nltk.corpus import stopwords

def check_alerts(df):
    # Ensure stopwords are loaded
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    # Calculate change in average sentiment
    delta = df["score"].iloc[-1] - df["score"].iloc[-2]
    alerts = []

    if abs(delta) > 0.2:
        alerts.append(f"Sudden sentiment change detected: {delta:+.2f}")

    # Combine all text
    words = " ".join(df["text"]).lower().split()

    # Filter out stopwords and short/common words
    filtered = [w for w in words if w not in stop_words and len(w) > 3 and w.isalpha()]

    # Get top 5 most frequent meaningful words
    if filtered:
        top_words = pd.Series(filtered).value_counts().head(5)
        alerts.append("Top trending words: " + ", ".join(top_words.index))
    else:
        alerts.append("No significant trending words found.")

    return alerts


#SLACK ALERT 
def send_slack_alert(message):
    if not SLACK_WEBHOOK_URL:
        print("(Slack not configured) ->", message)
        return
    requests.post(SLACK_WEBHOOK_URL, json={"text": message})

#VISUALIZATION
def plot_forecast(daily, forecast_df):
    plt.figure(figsize=(8,5))
    plt.plot(daily["date"], daily["score"], marker='o', label="Historical Sentiment")
    plt.plot(forecast_df["date"], forecast_df["forecast_score"], marker='x', linestyle='--', color='orange', label="Forecast")
    plt.axhline(0, color='gray', linestyle=':')
    plt.title("AI Market Sentiment Forecast Trend (Real Data)")
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_trend.png")
    plt.show()

#  MAIN 
def main():
    print("Fetching real data & analyzing trends...")
    df = fetch_articles("artificial intelligence", days=7)
    if df.empty:
        print("No articles found.")
        return

    df = analyze_sentiment(df)
    daily, forecast_df = train_and_forecast(df)
    alerts = check_alerts(df)

    trend = "📈 Uptrend" if forecast_df["forecast_score"].iloc[-1] > forecast_df["forecast_score"].iloc[0] else "📉 Downtrend"
    message = f"*AI Market Trend Forecast:* {trend}\nNext {len(forecast_df)} days: {forecast_df['forecast_score'].round(2).tolist()}\n" + "\n".join(alerts)

    print(message)
    send_slack_alert(message)
    plot_forecast(daily, forecast_df)

    # Word cloud for visualization
    text = " ".join(df["title"].dropna())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of AI News Headlines")
    plt.tight_layout()
    plt.savefig("wordcloud.png")
    plt.show()

if __name__ == "__main__":
    main()
