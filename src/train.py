import os
import json
import pandas as pd

from src.config import Paths, TrainConfig
from src.m5_data import build_merged_long, select_top_series
from src.features import build_supervised_features, feature_columns, add_calendar_features
from src.cv import make_walk_forward_folds
from src.metrics import mape, rmse
from src.models import (
    naive_last_value_forecast,
    fit_predict_prophet,
    fit_predict_xgb,
    fit_final_prophet,
    fit_final_xgb,
)
from src.registry import save_version

def ensure_dirs(paths: Paths):
    os.makedirs(paths.models_dir, exist_ok=True)
    os.makedirs(paths.reports_dir, exist_ok=True)

def evaluate_one_series(series_df: pd.DataFrame, cfg: TrainConfig) -> dict:
    """
    Returns CV metrics for naive, prophet, xgb and selects best by avg CV MAPE.
    """
    series_df = add_calendar_features(series_df)

    folds = make_walk_forward_folds(
        series_df, horizon_days=cfg.horizon_days, n_folds=cfg.n_folds, step_days=cfg.fold_step_days
    )

    feat_df = build_supervised_features(series_df)

    results = {"naive": [], "prophet": [], "xgb": []}
    rmses = {"naive": [], "prophet": [], "xgb": []}

    for fold in folds:
        train_mask_raw = series_df["date"] <= fold.train_end
        test_mask_raw = (series_df["date"] >= fold.test_start) & (series_df["date"] <= fold.test_end)

        raw_train = series_df.loc[train_mask_raw].copy()
        raw_test = series_df.loc[test_mask_raw].copy()

        if len(raw_test) != cfg.horizon_days or len(raw_train) < 60:
            continue

        # ----- Naive baseline -----
        naive_pred = naive_last_value_forecast(raw_train["sales"].values, cfg.horizon_days)
        results["naive"].append(mape(raw_test["sales"].values, naive_pred))
        rmses["naive"].append(rmse(raw_test["sales"].values, naive_pred))

        # ----- Prophet -----
        prophet_pred = fit_predict_prophet(raw_train, raw_test)
        results["prophet"].append(mape(raw_test["sales"].values, prophet_pred))
        rmses["prophet"].append(rmse(raw_test["sales"].values, prophet_pred))

        # ----- XGBoost -----
        # For XGB we evaluate on feature DF (drops early rows due to lags)
        tr_mask_feat = feat_df["date"] <= fold.train_end
        te_mask_feat = (feat_df["date"] >= fold.test_start) & (feat_df["date"] <= fold.test_end)
        tr_feat = feat_df.loc[tr_mask_feat].copy()
        te_feat = feat_df.loc[te_mask_feat].copy()

        if len(te_feat) != cfg.horizon_days or len(tr_feat) < 200:
            continue

        xgb_pred = fit_predict_xgb(tr_feat, te_feat, cfg.xgb_params)
        results["xgb"].append(mape(te_feat["sales"].values, xgb_pred))
        rmses["xgb"].append(rmse(te_feat["sales"].values, xgb_pred))

    def avg(x): 
        return float(pd.Series(x).mean()) if len(x) else float("inf")

    cv_mape = {k: avg(v) for k, v in results.items()}
    cv_rmse = {k: avg(v) for k, v in rmses.items()}

    best_model = min(cv_mape, key=cv_mape.get)

    return {
        "cv_mape": cv_mape,
        "cv_rmse": cv_rmse,
        "best_model": best_model,
        "n_folds_used": {k: len(v) for k, v in results.items()}
    }

def fit_and_save_best(series_id: str, series_df: pd.DataFrame, cfg: TrainConfig, paths: Paths, eval_summary: dict):
    series_df = series_df.sort_values("date").reset_index(drop=True)
    series_df = add_calendar_features(series_df)

    best = eval_summary["best_model"]

    metadata = {
        "best_model": best,
        "cv_mape": eval_summary["cv_mape"],
        "cv_rmse": eval_summary["cv_rmse"],
        "horizon_days": cfg.horizon_days,
        "features": feature_columns(),
    }

    if best == "prophet":
        model = fit_final_prophet(series_df)
    elif best == "xgb":
        feat_df = build_supervised_features(series_df)
        model = fit_final_xgb(feat_df, cfg.xgb_params)
        metadata["xgb_params"] = cfg.xgb_params
    else:
        model = fit_final_prophet(series_df)
        metadata["best_model"] = "prophet_fallback_from_naive"
        metadata["note"] = "Naive won CV; saved Prophet for deployable inference."

    return save_version(paths.models_dir, series_id, model, metadata)

def main():
    paths = Paths()
    cfg = TrainConfig()
    ensure_dirs(paths)

    print("Loading & merging M5 data (this may take a bit)...")
    df = build_merged_long(paths)

    print("Selecting top series...")
    top_ids = select_top_series(df, cfg.top_n_series)
    df_top = df[df["id"].isin(top_ids)].copy()

    df_top = (
        df_top.groupby(["id", "date", "state_id", "wday", "month", "year",
                        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                        "snap_CA", "snap_TX", "snap_WI"], as_index=False)
              .agg({"sales": "sum", "sell_price": "mean"})
              .sort_values(["id", "date"])
    )

    report_rows = []

    for series_id, sdf in df_top.groupby("id"):
        sdf = sdf.sort_values("date").reset_index(drop=True)

        if sdf["sales"].sum() < 500:
            continue

        print(f"\n=== Series: {series_id} ===")
        eval_summary = evaluate_one_series(sdf, cfg)
        print("CV MAPE:", eval_summary["cv_mape"], "Best:", eval_summary["best_model"])

        saved = fit_and_save_best(series_id, sdf, cfg, paths, eval_summary)

        report_rows.append({
            "series_id": series_id,
            "best_model": eval_summary["best_model"],
            "naive_cv_mape": eval_summary["cv_mape"]["naive"],
            "prophet_cv_mape": eval_summary["cv_mape"]["prophet"],
            "xgb_cv_mape": eval_summary["cv_mape"]["xgb"],
            "naive_cv_rmse": eval_summary["cv_rmse"]["naive"],
            "prophet_cv_rmse": eval_summary["cv_rmse"]["prophet"],
            "xgb_cv_rmse": eval_summary["cv_rmse"]["xgb"],
            "saved_version": saved["version"],
            "saved_path": saved["path"],
            "folds_used_naive": eval_summary["n_folds_used"]["naive"],
            "folds_used_prophet": eval_summary["n_folds_used"]["prophet"],
            "folds_used_xgb": eval_summary["n_folds_used"]["xgb"],
        })

    report_df = pd.DataFrame(report_rows).sort_values("xgb_cv_mape", ascending=True)
    out_csv = os.path.join(paths.reports_dir, "model_comparison.csv")
    report_df.to_csv(out_csv, index=False)

    print("\nSaved report:", out_csv)
    print(report_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()