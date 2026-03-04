import pandas as pd
import numpy as np

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Basic time features
    out["dow"] = out["date"].dt.dayofweek
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)

    # Events (binary flags)
    out["has_event_1"] = out["event_name_1"].notna().astype(int)
    out["has_event_2"] = out["event_name_2"].notna().astype(int)

    # One “has_any_event”
    out["has_event"] = ((out["has_event_1"] == 1) | (out["has_event_2"] == 1)).astype(int)

    out["snap"] = 0
    out.loc[out["state_id"] == "CA", "snap"] = out.loc[out["state_id"] == "CA", "snap_CA"]
    out.loc[out["state_id"] == "TX", "snap"] = out.loc[out["state_id"] == "TX", "snap_TX"]
    out.loc[out["state_id"] == "WI", "snap"] = out.loc[out["state_id"] == "WI", "snap_WI"]
    out["snap"] = out["snap"].fillna(0).astype(int)

    return out

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sell_price"] = out["sell_price"].astype(float)

    # Price change features (per series)
    out["price_change_1"] = out.groupby("id")["sell_price"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    out["price_rolling_mean_7"] = out.groupby("id")["sell_price"].transform(lambda s: s.rolling(7).mean())
    out["price_rolling_mean_28"] = out.groupby("id")["sell_price"].transform(lambda s: s.rolling(28).mean())

    return out

def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
   
    out = df.copy()
    g = out.groupby("id")["sales"]

    # Lags
    for lag in [1, 7, 14, 28]:
        out[f"lag_{lag}"] = g.shift(lag)

    out["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    out["roll_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    out["roll_std_7"] = g.transform(lambda s: s.shift(1).rolling(7).std())
    out["roll_std_28"] = g.transform(lambda s: s.shift(1).rolling(28).std())

    return out

def build_supervised_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_calendar_features(out)
    out = add_price_features(out)
    out = add_lag_rolling_features(out)

    needed = ["id", "date", "sales"] + feature_columns()
    out = out.dropna(subset=needed).reset_index(drop=True)
    return out

def feature_columns() -> list[str]:
    return [
        # time
        "wday", "month", "year", "dow", "weekofyear",
        # events + snap
        "has_event", "snap",
        # price
        "sell_price", "price_change_1", "price_rolling_mean_7", "price_rolling_mean_28",
        # lags/rolling
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_28", "roll_std_7", "roll_std_28"
    ]