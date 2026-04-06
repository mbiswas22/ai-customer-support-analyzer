from textblob import TextBlob

# Category classification
def classify(text):
    text = text.lower()
    if "payment" in text or "refund" in text:
        return "Billing"
    elif "error" in text or "crash" in text:
        return "Technical"
    elif "account" in text or "login" in text:
        return "Account"
    return "General"


# Sentiment analysis
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    return "Positive"


# Urgency detection
def is_urgent(text):
    urgent_words = ["urgent", "asap", "immediately", "not working"]
    return any(word in text.lower() for word in urgent_words)


def process_predictions(df):
    df["category"] = df["text"].apply(classify)
    df["sentiment"] = df["text"].apply(get_sentiment)
    df["urgent"] = df["text"].apply(is_urgent)
    return df