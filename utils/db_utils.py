import json
import pandas as pd
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from utils.db_server import SERVER_PORT

BASE_URL = f"http://localhost:{SERVER_PORT}"

def _to_df(data: list) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "urgent" in df.columns:
        df["urgent"] = df["urgent"].map(lambda x: str(x).strip().lower() in ("true", "1"))
    if "response_time_hours" in df.columns:
        df["response_time_hours"] = pd.to_numeric(df["response_time_hours"], errors="coerce")
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def _get(path: str, params: dict = None) -> any:
    url = f"{BASE_URL}{path}"
    if params:
        url += "?" + urlencode({k: v for k, v in params.items() if v is not None})
    with urlopen(url) as r:
        return json.loads(r.read())


def _post(path: str, data: list) -> dict:
    body = json.dumps(data).encode()
    req = Request(f"{BASE_URL}{path}", data=body, headers={"Content-Type": "application/json"})
    with urlopen(req) as r:
        return json.loads(r.read())


def init_db():
    pass  # Table is created on first /tickets/load POST


def insert_tickets(df: pd.DataFrame):
    records = json.loads(df.to_json(orient="records", date_format="iso"))
    _post("/tickets/load", records)


def get_all_tickets() -> pd.DataFrame:
    return _to_df(_get("/tickets/all"))


def get_filtered_tickets(
    category=None, sentiment=None, priority=None,
    urgent_only=False, start_date=None, end_date=None
) -> pd.DataFrame:
    params = {
        "category": category,
        "sentiment": sentiment,
        "priority": priority,
        "urgent_only": "1" if urgent_only else None,
        "start_date": str(start_date) if start_date else None,
        "end_date": str(end_date) if end_date else None,
    }
    return _to_df(_get("/tickets", params))


def get_distinct_values(column: str) -> list:
    return _get("/tickets/distinct", {"column": column})


def get_ticket_texts() -> list:
    return _get("/tickets/texts")


def get_daily_counts(category=None) -> pd.DataFrame:
    params = {"category": category} if category else {}
    df = pd.DataFrame(_get("/tickets/daily_counts", params))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_date_range() -> tuple:
    data = _get("/tickets/date_range")
    return pd.to_datetime(data["min"]), pd.to_datetime(data["max"])
