import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.db_utils import get_daily_counts


def forecast_tickets(df=None, days_ahead=7, category=None):
    series = get_daily_counts(category)
    if len(series) < 3:
        return None, None

    series["ordinal"] = series["date"].map(pd.Timestamp.toordinal)
    X = series[["ordinal"]].values
    y = series["count"].values

    model = LinearRegression().fit(X, y)

    last_date = series["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead)
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    predictions = np.maximum(0, model.predict(future_ordinals).round().astype(int))

    forecast_df = pd.DataFrame({"date": future_dates, "predicted_tickets": predictions})
    return series, forecast_df
