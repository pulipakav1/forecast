import time
from functools import lru_cache
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import Paths
from src.m5_data import build_merged_long
from src.features import add_calendar_features, build_supervised_features, feature_columns
from src.registry import load_latest

app = FastAPI(title="M5 Forecast API (DS Project)", version="1.0")

REQUESTS = 0
TOTAL_LAT_MS = 0.0

class PredictRequest(BaseModel):
    series_id: str = Field(..., description="M5 series id, e.g., FOODS_1_001_CA_1_validation")
    periods: int = Field(default=28, ge=1, le=56)

@lru_cache(maxsize=1)
def get_paths():
    return Paths()

@lru_cache(maxsize=1)
def get_data():
    """
    Cache merged long data in memory.
    For very low RAM machines, you can change this to lazy-load per series.
    """
    paths = get_paths()
    df = build_merged_long(paths)
    # keep only what we need
    df = df[["id", "date", "sales", "sell_price", "state_id",
             "wday", "month", "year",
             "event_name_1", "event_type_1", "event_name_2", "event_type_2",
             "snap_CA", "snap_TX", "snap_WI"]].copy()
    df = (
        df.groupby(["id", "date", "state_id", "wday", "month", "year",
                    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                    "snap_CA", "snap_TX", "snap_WI"], as_index=False)
          .agg({"sales": "sum", "sell_price": "mean"})
          .sort_values(["id", "date"])
    )
    return df

@lru_cache(maxsize=64)
def get_model(series_id: str):
    paths = get_paths()
    return load_latest(paths.models_dir, series_id)

def _future_frame_for_series(df_series: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Build future rows for Prophet using last known features.
    For M5, calendar/prices are only known within the dataset range.
    So we forecast within available future dates if present.
    """
    last_date = df_series["date"].max()
    # We will create future dates using daily frequency
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")

    # Build a frame with placeholders; we’ll fill regressors conservatively
    fut = pd.DataFrame({"date": future_dates})

    # Carry-forward last known sell_price (simple, realistic assumption)
    last_price = float(df_series["sell_price"].iloc[-1])
    fut["sell_price"] = last_price

    # Use calendar-derived features from date
    fut["wday"] = fut["date"].dt.day_name().map({
        "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
        "Friday": 5, "Saturday": 6, "Sunday": 7
    }).astype(int)
    fut["month"] = fut["date"].dt.month.astype(int)
    fut["year"] = fut["date"].dt.year.astype(int)

    # No future event/snap info here (set to 0). If you want, we can merge from calendar.csv for true future flags.
    fut["event_name_1"] = None
    fut["event_type_1"] = None
    fut["event_name_2"] = None
    fut["event_type_2"] = None
    fut["snap_CA"] = 0
    fut["snap_TX"] = 0
    fut["snap_WI"] = 0

    # Carry state_id from series
    fut["state_id"] = df_series["state_id"].iloc[-1]
    fut["sales"] = np.nan  # unknown
    fut = add_calendar_features(fut)
    return fut

def _predict_prophet(model, df_series: pd.DataFrame, periods: int):
    fut = _future_frame_for_series(df_series, periods)
    df_te = fut.rename(columns={"date": "ds"}).copy()

    forecast = model.predict(df_te[["ds", "sell_price", "snap", "has_event"]])
    yhat = forecast["yhat"].to_numpy(dtype=float)
    return fut["date"].astype(str).tolist(), yhat.tolist()

def _predict_xgb_recursive(model, df_series: pd.DataFrame, periods: int):
    """
    Recursive forecasting using lag features.
    We assume sell_price stays constant and events/snap=0 for future unless you merge calendar.csv.
    """
    # Build supervised df to get the last available feature row
    hist = df_series.copy()
    hist = add_calendar_features(hist)
    feat_hist = build_supervised_features(hist)

    feats = feature_columns()
    if feat_hist.empty:
        raise HTTPException(status_code=400, detail="Not enough history for lag features.")

    # start from last known date
    last_date = hist["date"].max()
    state_id = hist["state_id"].iloc[-1]
    last_price = float(hist["sell_price"].iloc[-1])

    sales_series = hist.sort_values("date")["sales"].to_list()

    out_dates = []
    out_preds = []

    for i in range(periods):
        dt = last_date + pd.Timedelta(days=i+1)
        out_dates.append(str(dt.date()))

        # Build a one-row feature frame “as if it were observed”
        row = {
            "id": hist["id"].iloc[-1] if "id" in hist.columns else "unknown",
            "date": dt,
            "state_id": state_id,
            "sell_price": last_price,
            "wday": int(dt.dayofweek + 1),
            "month": int(dt.month),
            "year": int(dt.year),
            "event_name_1": None,
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": 0,
            "snap_TX": 0,
            "snap_WI": 0,
            "sales": 0.0  # placeholder
        }
        tmp = pd.DataFrame([row])
        tmp = add_calendar_features(tmp)
        # Add price features expected by pipeline
        tmp["price_change_1"] = 0.0
        tmp["price_rolling_mean_7"] = last_price
        tmp["price_rolling_mean_28"] = last_price

        def lag(k): 
            return float(sales_series[-k]) if len(sales_series) >= k else float(sales_series[-1])

        tmp["lag_1"] = lag(1)
        tmp["lag_7"] = lag(7)
        tmp["lag_14"] = lag(14)
        tmp["lag_28"] = lag(28)

        # Rolling stats
        def roll_mean(w):
            vals = sales_series[-w:] if len(sales_series) >= w else sales_series
            return float(np.mean(vals))

        def roll_std(w):
            vals = sales_series[-w:] if len(sales_series) >= w else sales_series
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        tmp["roll_mean_7"] = roll_mean(7)
        tmp["roll_mean_28"] = roll_mean(28)
        tmp["roll_std_7"] = roll_std(7)
        tmp["roll_std_28"] = roll_std(28)

        # time extras
        tmp["dow"] = tmp["date"].dt.dayofweek
        tmp["weekofyear"] = tmp["date"].dt.isocalendar().week.astype(int)

        X = tmp[feats].to_numpy()
        pred = float(model.predict(X)[0])

        out_preds.append(pred)
        sales_series.append(pred)  # recursive update

    return out_dates, out_preds

@app.post("/predict")
def predict(req: PredictRequest):
    global REQUESTS, TOTAL_LAT_MS
    start = time.perf_counter()

    df = get_data()
    df_series = df[df["id"] == req.series_id].sort_values("date").copy()
    if df_series.empty:
        raise HTTPException(status_code=404, detail="series_id not found in data")

    model, meta = get_model(req.series_id)
    model_type = meta.get("best_model")

    if model_type == "prophet":
        dates, preds = _predict_prophet(model, df_series, req.periods)
    elif model_type == "xgb":
        dates, preds = _predict_xgb_recursive(model, df_series, req.periods)
    else:
        # fallback
        dates, preds = _predict_prophet(model, df_series, req.periods)

    lat_ms = (time.perf_counter() - start) * 1000.0
    REQUESTS += 1
    TOTAL_LAT_MS += lat_ms

    return {
        "series_id": req.series_id,
        "model_type": model_type,
        "periods": req.periods,
        "latency_ms": round(lat_ms, 2),
        "forecast": [{"date": d, "yhat": float(y)} for d, y in zip(dates, preds)]
    }

@app.get("/metrics")
def metrics():
    avg = (TOTAL_LAT_MS / REQUESTS) if REQUESTS else 0.0
    return {"requests": REQUESTS, "avg_latency_ms": round(avg, 2)}