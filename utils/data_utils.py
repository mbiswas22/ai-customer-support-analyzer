import pandas as pd
from utils.db_utils import init_db, insert_tickets


def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["text_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    return df


def save_to_db(df: pd.DataFrame):
    """Persist the fully processed DataFrame into SQLite."""
    init_db()
    insert_tickets(df)
