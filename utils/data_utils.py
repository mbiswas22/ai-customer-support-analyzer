import pandas as pd

def load_data(file):
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    return df


def add_features(df):
    df["text_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    return df

def filter_data(df, category=None, sentiment=None, urgent_only=False, start_date=None, end_date=None):
    filtered = df.copy()
    if category and category != "All":
        filtered = filtered[filtered["category"] == category]
    if sentiment and sentiment != "All":
        filtered = filtered[filtered["sentiment"] == sentiment]
    if urgent_only:
        filtered = filtered[filtered["urgent"] == True]
    if start_date:
        filtered = filtered[filtered["created_at"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["created_at"] <= pd.to_datetime(end_date)]
    return filtered