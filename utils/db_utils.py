import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    pass  # Table is created dynamically from DataFrame schema


def insert_tickets(df: pd.DataFrame):
    """Replace all tickets with the newly uploaded CSV data."""
    with _conn() as con:
        df.to_sql("tickets", con, if_exists="replace", index=False)


# ------------------------------------------------------------------
# Query APIs
# ------------------------------------------------------------------

def get_all_tickets() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql("SELECT * FROM tickets", con, parse_dates=["created_at"])


def get_filtered_tickets(
    category=None, sentiment=None, priority=None,
    urgent_only=False, start_date=None, end_date=None
) -> pd.DataFrame:
    clauses, params = [], []

    if category and category != "All":
        clauses.append("category = ?")
        params.append(category)
    if sentiment and sentiment != "All":
        clauses.append("sentiment = ?")
        params.append(sentiment)
    if priority and priority != "All":
        clauses.append("priority = ?")
        params.append(priority)
    if urgent_only:
        clauses.append("urgent = 1")
    if start_date:
        clauses.append("created_at >= ?")
        params.append(str(start_date))
    if end_date:
        clauses.append("created_at <= ?")
        params.append(str(end_date))

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM tickets {where}"

    with _conn() as con:
        return pd.read_sql(sql, con, params=params, parse_dates=["created_at"])


def get_distinct_values(column: str) -> list:
    with _conn() as con:
        rows = con.execute(f"SELECT DISTINCT {column} FROM tickets ORDER BY {column}").fetchall()
    return [r[0] for r in rows if r[0] is not None]


def get_ticket_texts() -> list:
    with _conn() as con:
        rows = con.execute("SELECT text FROM tickets").fetchall()
    return [r[0] for r in rows]


def get_daily_counts(category=None) -> pd.DataFrame:
    if category and category != "All":
        sql = """
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM tickets WHERE category = ?
            GROUP BY DATE(created_at) ORDER BY date
        """
        params = [category]
    else:
        sql = """
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM tickets GROUP BY DATE(created_at) ORDER BY date
        """
        params = []

    with _conn() as con:
        df = pd.read_sql(sql, con, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_date_range() -> tuple:
    with _conn() as con:
        row = con.execute("SELECT MIN(created_at), MAX(created_at) FROM tickets").fetchone()
    return pd.to_datetime(row[0]), pd.to_datetime(row[1])
