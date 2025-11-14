import os
import pandas as pd
import requests
from dotenv import load_dotenv

#Here we are Loading Slack Key
load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

#Sending the Slack Helper 
def send_slack_alert(message: str):
    """Send a simple message to Slack using incoming webhook."""
    if not SLACK_WEBHOOK_URL:
        print("Slack webhook URL missing.")
        return
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=10)
        if resp.status_code == 200:
            print("Slack message sent")
        else:
            print(f"Slack error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Slack exception: {e}")

#Main function
def main():
    print("Task 2: Sending Sentiment Alerts to Slack...")
    if not os.path.exists("ai_news_sentiment.csv"):
        print("CSV not found. Run Task 1 first.")
        return

    df = pd.read_csv("ai_news_sentiment.csv")

    for sentiment, emoji in [("Positive"), ("Neutral"), ("Negative")]:
        subset = df[df["sentiment"] == sentiment]
        if subset.empty:
            send_slack_alert(f"{emoji} No {sentiment.lower()} AI news found today.")
            continue

        # Building a combined message per sentiment
        msg_lines = [f"*{emoji} {sentiment} AI News Alerts*"]
        for _, row in subset.iterrows():
            msg_lines.append(
                f"• *{row['title']}* ({row['source']})\n{row['description']}\n🔗 {row['url']}"
            )
        full_msg = "\n\n".join(msg_lines)
        send_slack_alert(full_msg)

    print("All sentiment alerts sent to Slack.")

if __name__ == "__main__":
    main()
