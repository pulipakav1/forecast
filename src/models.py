import pandas as pd
import numpy as np
import xgboost as xgb

from src.features import feature_columns

def naive_last_value_forecast(train_y: np.ndarray, horizon: int) -> np.ndarray:
    last = float(train_y[-1])
    return np.full(shape=(horizon,), fill_value=last, dtype=float)

def fit_predict_xgb(train_feat: pd.DataFrame, test_feat: pd.DataFrame, params: dict) -> np.ndarray:
    feats = feature_columns()

    X_train = train_feat[feats].to_numpy()
    y_train = train_feat["sales"].to_numpy(dtype=float)

    X_test = test_feat[feats].to_numpy()

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    return model.predict(X_test).astype(float)



def fit_final_xgb(full_feat: pd.DataFrame, params: dict):
    feats = feature_columns()

    X = full_feat[feats].to_numpy()
    y = full_feat["sales"].to_numpy(dtype=float)

    model = xgb.XGBRegressor(**params)
    model.fit(X, y)

    return model