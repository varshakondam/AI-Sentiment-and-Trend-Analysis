"""
AI News Sentiment & Market Forecast Pro — Final Version (Prophet + Slack)
Run using:
    streamlit run app.py
Ensure .env contains:
    NEWSAPI_KEY=your_newsapi_key
    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
Dependencies:
    pip install streamlit pandas plotly wordcloud prophet python-dotenv matplotlib nltk
"""

# -----------------------------------------------------------
# 🔧 Imports & Setup
# -----------------------------------------------------------
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from ai_news_sentiment_module1 import fetch_articles, analyze_sentiment, save_results_and_visuals
import requests
from prophet import Prophet

# -----------------------------------------------------------
# 🌐 Load API Keys
# -----------------------------------------------------------
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# -----------------------------------------------------------
# 🎨 Streamlit Page Configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI News Sentiment & Market Forecast Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #001f3f, #000a1a, #000);
    color: white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #002b4d, #000a1a);
}
h1, h2, h3 {
    color: #00BFFF !important;
}
.stButton>button {
    background: linear-gradient(90deg, #00BFFF, #1E90FF);
    color: white;
    border-radius: 8px;
    font-weight: bold;
    box-shadow: 0px 0px 10px #00BFFF;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #FFD700, #FFA500);
    color: black;
}
hr {border: 1px solid #00BFFF;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# ⚙️ Sidebar Controls
# -----------------------------------------------------------
st.sidebar.header("⚙️ Controls")
query = st.sidebar.text_input("Topic", "artificial intelligence")
days = st.sidebar.slider("Days to Fetch", 3, 30, 7)
show_wordcloud = st.sidebar.checkbox("Show Word Cloud", True)
show_forecast = st.sidebar.checkbox("Run Forecast", True)
send_slack = st.sidebar.checkbox("Send Slack Alerts", False)
run_btn = st.sidebar.button("🚀 Run Full Analysis")

# -----------------------------------------------------------
# 🔔 Slack Alert Function
# -----------------------------------------------------------
def send_slack_alert(message_text: str) -> bool:
    """Send a formatted message to Slack via Incoming Webhook."""
    if not SLACK_WEBHOOK_URL:
        st.warning("⚠️ Slack webhook missing in .env")
        return False

    payload = {"text": message_text}
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        if r.status_code == 200:
            st.toast("✅ Slack alert delivered", icon="✅")
            return True
        else:
            st.error(f"Slack responded {r.status_code}: {r.text}")
            return False
    except Exception as e:
        st.error(f"Slack send error: {e}")
        return False

# -----------------------------------------------------------
# ⚙️ Helper Functions
# -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def prepare_prophet_data(df):
    df["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.date
    daily = df.groupby("date")["score"].mean().reset_index().rename(columns={"date": "ds", "score": "y"})
    return daily

def run_prophet_forecast(df, periods=7):
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# -----------------------------------------------------------
# 🧠 HEADER
# -----------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🧠 AI News Sentiment & Market Forecast Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Real-time sentiment analytics and forecasting for AI industry trends.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🚀 MAIN EXECUTION
# -----------------------------------------------------------
if run_btn:
    with st.spinner("Fetching latest AI news & analyzing sentiment..."):
        df = fetch_articles(query=query, days=days)
        df = analyze_sentiment(df)
        save_results_and_visuals(df)
    st.success(f"✅ Analyzed {len(df)} articles on *{query.title()}*")

    # ------------------- METRICS -------------------
    pos = (df["sentiment"] == "Positive").sum()
    neu = (df["sentiment"] == "Neutral").sum()
    neg = (df["sentiment"] == "Negative").sum()
    total = len(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("😀 Positive", pos, f"{(pos/total)*100:.1f}%")
    with col2:
        st.metric("😐 Neutral", neu, f"{(neu/total)*100:.1f}%")
    with col3:
        st.metric("☹️ Negative", neg, f"{(neg/total)*100:.1f}%")

    # ------------------- SENTIMENT PIE -------------------
    st.markdown("### 🎯 Sentiment Distribution")
    fig = px.pie(
        df, names="sentiment", color="sentiment",
        color_discrete_map={"Positive": "#00FF99", "Neutral": "#CCCCCC", "Negative": "#FF4C4C"},
        title="Sentiment Composition", hole=0.3
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------- TREND CHART -------------------
    st.markdown("### 📈 Sentiment Over Time")
    df["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.date
    trend = df.groupby("date")["score"].mean().reset_index()
    fig2 = px.line(trend, x="date", y="score", markers=True,
                   title="Average Sentiment Trend", line_shape="spline",
                   color_discrete_sequence=["#00BFFF"])
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------- SOURCE INSIGHTS -------------------
    st.markdown("### 🏢 Top Sources Overview")
    top_sources = df["source"].value_counts().nlargest(10)
    fig3 = px.bar(
        x=top_sources.index, y=top_sources.values,
        color=top_sources.values, color_continuous_scale="Blues",
        title="Top 10 Sources by Article Count"
    )
    st.plotly_chart(fig3, use_container_width=True)

    by_source = df.groupby(["source", "sentiment"]).size().reset_index(name="count")
    fig4 = px.bar(
        by_source, x="source", y="count", color="sentiment",
        color_discrete_map={"Positive": "#00FF99", "Neutral": "#CCCCCC", "Negative": "#FF4C4C"},
        title="Sentiment Breakdown by Source"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ------------------- WORD CLOUD -------------------
    if show_wordcloud:
        st.markdown("### ☁️ Word Cloud of Headlines")
        text = " ".join(df["title"].dropna())
        wc = WordCloud(width=900, height=400, background_color="black", colormap="cool").generate(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    # ------------------- FORECASTING SECTION -------------------
    if show_forecast:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## 🔮 Sentiment Forecasting (Prophet Model)")

        daily = prepare_prophet_data(df)
        if len(daily) < 3:
            st.warning("⚠️ Not enough data for Prophet forecast (need ≥3 days).")
        else:
            model, forecast = run_prophet_forecast(daily)
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], mode="lines+markers",
                                      name="Historical", line=dict(color="#00BFFF")))
            fig5.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines",
                                      name="Forecast", line=dict(color="#FFD700", dash="dash")))
            fig5.add_hline(y=0, line_dash="dot", line_color="white")
            fig5.update_layout(title="Prophet Sentiment Forecast", template="plotly_dark")
            st.plotly_chart(fig5, use_container_width=True)

            future_forecast = forecast.tail(7)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            avg_future = future_forecast["yhat"].mean()
            latest = daily["y"].iloc[-1]
            trend_symbol = "📈" if avg_future > latest else "📉" if avg_future < latest else "➡️"
            trend_type = f"{trend_symbol} {'Uptrend' if avg_future > latest else 'Downtrend' if avg_future < latest else 'Stable'}"

            st.metric("Current Avg", f"{latest:.3f}")
            st.metric("Forecast Avg", f"{avg_future:.3f}")
            st.success(f"Predicted Trend: {trend_type}")

            # Compose summary
            summary = f"""
**📊 AI Sentiment Forecast Summary**
> **Trend:** {trend_type}  
> **Current Avg:** {latest:.3f}  
> **Forecast Avg (next 7 days):** {avg_future:.3f}  
> **Model:** Prophet  
> **Data Points:** {len(daily)}  
"""

            st.info(summary)

            # ------------------- SLACK ALERT -------------------
            if send_slack:
                st.info("📤 Sending forecast summary to Slack...")
                success = send_slack_alert(summary)
                if success:
                    st.success("✅ Slack message sent successfully!")
                    st.balloons()

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center;color:gray;'>© 2025 AI Sentiment Intelligence Pro Dashboard</h5>", unsafe_allow_html=True)
