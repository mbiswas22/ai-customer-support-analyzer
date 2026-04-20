import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv
from textblob import TextBlob

load_dotenv()

# Weighted critical keywords — higher weight = more severe
_CRITICAL_KEYWORDS = {
    "data loss": 10, "breach": 10, "hacked": 10, "security": 8,
    "down": 7, "outage": 7, "not working": 6, "broken": 6,
    "urgent": 5, "asap": 5, "immediately": 5, "critical": 5,
    "refund": 4, "charge": 4, "overcharged": 4, "locked out": 4,
    "error": 3, "crash": 3, "failed": 3, "cannot": 3,
}

def _criticality_score(row):
    text = row["text"].lower()

    # 1. Keyword score (0–40 range)
    keyword_score = sum(w for kw, w in _CRITICAL_KEYWORDS.items() if kw in text)

    # 2. Sentiment score: more negative = higher (0–10 range)
    polarity = TextBlob(row["text"]).sentiment.polarity  # -1 to 1
    sentiment_score = (1 - polarity) * 5  # 0 (positive) to 10 (very negative)

    # 3. Response time score: longer wait = higher (0–10 range)
    response_time = row.get("response_time_hours", 0) or 0
    response_score = min(response_time / 10, 10)  # caps at 10

    # 4. Text length score: longer = more complex/frustrated (0–5 range)
    length_score = min(len(row["text"]) / 100, 5)

    return keyword_score + sentiment_score + response_score + length_score


def send_high_priority_alert(df):
    high_df = df[df["priority"] == "high"].copy()
    if high_df.empty:
        return 0

    high_df["_score"] = high_df.apply(_criticality_score, axis=1)
    top10 = high_df.nlargest(10, "_score")

    body = f"Top {len(top10)} most critical HIGH priority ticket(s):\n\n"
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        body += (
            f"#{rank} [Score: {row['_score']:.1f}] "
            f"[{row['category']} | {row['sentiment']}]\n"
            f"{row['text'][:200]}\n"
            f"{'─' * 60}\n"
        )

    message = Mail(
        from_email=os.getenv("ALERT_FROM_EMAIL"),
        to_emails=os.getenv("ALERT_TO_EMAIL"),
        subject=f"🚨 Top {len(top10)} Critical Support Tickets Detected",
        plain_text_content=body
    )
    SendGridAPIClient(os.getenv("SENDGRID_API_KEY")).send(message)
    return len(top10)
