import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb

from src.features import feature_columns

def naive_last_value_forecast(train_y: np.ndarray, horizon: int) -> np.ndarray:
    last = float(train_y[-1])
    return np.full(shape=(horizon,), fill_value=last, dtype=float)

def fit_predict_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    Prophet expects columns: ds, y. We also add regressors.
    """
    df_tr = train_df.rename(columns={"date": "ds", "sales": "y"}).copy()
    df_te = test_df.rename(columns={"date": "ds"}).copy()

    # Regressors used
    regressors = ["sell_price", "snap", "has_event"]

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.08,
        seasonality_prior_scale=10.0,
        interval_width=0.9,
    )

    for r in regressors:
        model.add_regressor(r)

    model.fit(df_tr[["ds", "y"] + regressors])

    forecast = model.predict(df_te[["ds"] + regressors])
    return forecast["yhat"].to_numpy(dtype=float)

def fit_predict_xgb(train_feat: pd.DataFrame, test_feat: pd.DataFrame, params: dict) -> np.ndarray:
    feats = feature_columns()
    X_train = train_feat[feats].to_numpy()
    y_train = train_feat["sales"].to_numpy(dtype=float)

    X_test = test_feat[feats].to_numpy()

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model.predict(X_test).astype(float)

def fit_final_prophet(full_df: pd.DataFrame) -> Prophet:
    df = full_df.rename(columns={"date": "ds", "sales": "y"}).copy()
    regressors = ["sell_price", "snap", "has_event"]

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.08,
        seasonality_prior_scale=10.0,
        interval_width=0.9,
    )
    for r in regressors:
        model.add_regressor(r)
    model.fit(df[["ds", "y"] + regressors])
    return model

def fit_final_xgb(full_feat: pd.DataFrame, params: dict):
    feats = feature_columns()
    X = full_feat[feats].to_numpy()
    y = full_feat["sales"].to_numpy(dtype=float)
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model