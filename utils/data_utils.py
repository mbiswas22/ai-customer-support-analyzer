import pandas as pd

def load_data(file):
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    return df


def add_features(df):
    df["text_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    return df