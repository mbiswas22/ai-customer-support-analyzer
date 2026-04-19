import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

load_dotenv()

def send_high_priority_alert(df):
    high_df = df[df["priority"] == "high"]
    if high_df.empty:
        return 0

    body = f"{len(high_df)} HIGH priority ticket(s) detected:\n\n"
    body += "\n---\n".join(
        f"[{row['category']} | {row['sentiment']}] {row['text'][:200]}"
        for _, row in high_df.iterrows()
    )

    message = Mail(
        from_email=os.getenv("ALERT_FROM_EMAIL"),
        to_emails=os.getenv("ALERT_TO_EMAIL"),
        subject=f"🚨 {len(high_df)} High Priority Support Ticket(s) Detected",
        plain_text_content=body
    )
    SendGridAPIClient(os.getenv("SENDGRID_API_KEY")).send(message)
    return len(high_df)
