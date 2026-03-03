from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    data_dir: str = "data/m5"
    sales_csv: str = "data/m5/sales_train_validation.csv"
    calendar_csv: str = "data/m5/calendar.csv"
    prices_csv: str = "data/m5/sell_prices.csv"
    models_dir: str = "models"
    reports_dir: str = "reports"

@dataclass(frozen=True)
class TrainConfig:
    top_n_series: int = 50

    # Forecast settings
    horizon_days: int = 28

    n_folds: int = 3
    fold_step_days: int = 28

    # XGBoost parameters (strong baseline)
    xgb_params: dict = None

    def __post_init__(self):
        if self.xgb_params is None:
            object.__setattr__(self, "xgb_params", {
                "n_estimators": 800,
                "learning_rate": 0.03,
                "max_depth": 8,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
            })