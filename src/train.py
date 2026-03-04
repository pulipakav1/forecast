import os
import pandas as pd

from src.config import Paths, TrainConfig
from src.m5_data import build_merged_long, select_top_series
from src.features import build_supervised_features, feature_columns, add_calendar_features
from src.cv import make_walk_forward_folds
from src.metrics import mape, rmse
from src.models import naive_last_value_forecast, fit_predict_xgb, fit_final_xgb
from src.registry import save_version


def ensure_dirs(paths: Paths) -> None:
    os.makedirs(paths.models_dir, exist_ok=True)
    os.makedirs(paths.reports_dir, exist_ok=True)


def _avg(values) -> float:
    return float(pd.Series(values).mean()) if values else float("inf")


def evaluate_one_series(series_df: pd.DataFrame, cfg: TrainConfig) -> dict:


    # ---- Sort + calendar features ----
    series_df = series_df.sort_values("date").reset_index(drop=True)
    series_df = add_calendar_features(series_df)

    # ---- Build feature dataframe FIRST ----
    feat_df = build_supervised_features(series_df)

    # ---- Generate folds aligned with feature data ----
    folds = make_walk_forward_folds(
        feat_df,
        horizon_days=cfg.horizon_days,
        n_folds=cfg.n_folds,
        step_days=cfg.fold_step_days,
    )

    results = {"naive": [], "xgb": []}
    rmses = {"naive": [], "xgb": []}

    for fold in folds:

      
        raw_train = series_df.loc[series_df["date"] <= fold.train_end].copy()
        raw_test = series_df.loc[
            (series_df["date"] >= fold.test_start)
            & (series_df["date"] <= fold.test_end)
        ].copy()

        if len(raw_test) < cfg.horizon_days:
            continue

        raw_test = raw_test.tail(cfg.horizon_days)

        naive_pred = naive_last_value_forecast(
            raw_train["sales"].to_numpy(),
            cfg.horizon_days,
        )

        results["naive"].append(
            mape(raw_test["sales"].to_numpy(dtype=float), naive_pred)
        )
        rmses["naive"].append(
            rmse(raw_test["sales"].to_numpy(dtype=float), naive_pred)
        )

   

        tr_feat = feat_df.loc[feat_df["date"] <= fold.train_end].copy()
        te_feat = feat_df.loc[
            (feat_df["date"] >= fold.test_start)
            & (feat_df["date"] <= fold.test_end)
        ].copy()

        if len(te_feat) < cfg.horizon_days:
            continue

        te_feat = te_feat.tail(cfg.horizon_days)

        xgb_pred = fit_predict_xgb(tr_feat, te_feat, cfg.xgb_params)

        results["xgb"].append(
            mape(te_feat["sales"].to_numpy(dtype=float), xgb_pred)
        )
        rmses["xgb"].append(
            rmse(te_feat["sales"].to_numpy(dtype=float), xgb_pred)
        )

    def _avg(values):
        return float(pd.Series(values).mean()) if values else float("inf")

    cv_mape = {k: _avg(v) for k, v in results.items()}
    cv_rmse = {k: _avg(v) for k, v in rmses.items()}

    best_model = min(cv_mape, key=cv_mape.get)

    return {
        "cv_mape": cv_mape,
        "cv_rmse": cv_rmse,
        "best_model": best_model,
        "n_folds_used": {k: len(v) for k, v in results.items()},
    }


def fit_and_save_best(
    series_id: str,
    series_df: pd.DataFrame,
    cfg: TrainConfig,
    paths: Paths,
    eval_summary: dict,
) -> dict:
  
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

    feat_df = build_supervised_features(series_df)
    if feat_df.empty:
        raise ValueError(
            f"Feature frame is empty for {series_id}. "
            
        )
    model = fit_final_xgb(feat_df, cfg.xgb_params)
    metadata["xgb_params"] = cfg.xgb_params

    if best != "xgb":
        metadata["best_model"] = "xgb_fallback_from_naive"
        metadata["note"] = "Naive won CV; saved XGB for deployable inference."

    return save_version(paths.models_dir, series_id, model, metadata)


def main() -> None:
    paths = Paths()
    cfg = TrainConfig()
    ensure_dirs(paths)

    print("Loading & merging M5 data (this may take a bit)...")
    df = build_merged_long(paths)

    print("Selecting top series...")
    top_ids = select_top_series(df, cfg.top_n_series)
    df_top = df[df["id"].isin(top_ids)].copy()

    # Aggregate (defensive) + keep required calendar fields
    df_top = (
        df_top.groupby(
            [
                "id",
                "date",
                "state_id",
                "wday",
                "month",
                "year",
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ],
            as_index=False,
            dropna=False,
        )
        .agg({"sales": "sum", "sell_price": "mean"})
        .sort_values(["id", "date"])
        .reset_index(drop=True)
    )
    df_top["sell_price"] = (
        df_top.groupby("id")["sell_price"].transform(lambda s: s.ffill().bfill()).fillna(0.0)
    )

    report_rows = []

    for series_id, sdf in df_top.groupby("id"):
        sdf = sdf.sort_values("date").reset_index(drop=True)

        if float(sdf["sales"].sum()) < 500:
            continue

        print(f"\nSeries: {series_id}")
        eval_summary = evaluate_one_series(sdf, cfg)
        print(
            "CV MAPE:", eval_summary["cv_mape"],
            "Folds used:", eval_summary["n_folds_used"],
            "Best:", eval_summary["best_model"],
        )

        saved = fit_and_save_best(series_id, sdf, cfg, paths, eval_summary)

        report_rows.append(
            {
                "series_id": series_id,
                "best_model": eval_summary["best_model"],
                "naive_cv_mape": eval_summary["cv_mape"]["naive"],
                "xgb_cv_mape": eval_summary["cv_mape"]["xgb"],
                "naive_cv_rmse": eval_summary["cv_rmse"]["naive"],
                "xgb_cv_rmse": eval_summary["cv_rmse"]["xgb"],
                "saved_version": saved["version"],
                "saved_path": saved["path"],
                "folds_used_naive": eval_summary["n_folds_used"]["naive"],
                "folds_used_xgb": eval_summary["n_folds_used"]["xgb"],
            }
        )

    if not report_rows:
        print("\nNo series produced a valid CV run.")
        return

    report_df = pd.DataFrame(report_rows).sort_values("xgb_cv_mape", ascending=True)
    out_csv = os.path.join(paths.reports_dir, "model_comparison.csv")
    report_df.to_csv(out_csv, index=False)

    print("\nSaved report:", out_csv)
    print(report_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()