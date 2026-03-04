from pathlib import Path
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class Paths:
    data_dir: str = "data/m5"
    sales_csv: str = "data/m5/sales_train_validation.csv"
    calendar_csv: str = "data/m5/calendar.csv"
    prices_csv: str = "data/m5/sell_prices.csv"
    models_dir: str = "models"
    reports_dir: str = "reports"

    @classmethod
    def from_root(cls, root: Union[str, Path]) -> "Paths":
        """Paths with all entries resolved against project root (for notebooks)."""
        r = Path(root)
        return cls(
            data_dir=str(r / "data" / "m5"),
            sales_csv=str(r / "data" / "m5" / "sales_train_validation.csv"),
            calendar_csv=str(r / "data" / "m5" / "calendar.csv"),
            prices_csv=str(r / "data" / "m5" / "sell_prices.csv"),
            models_dir=str(r / "models"),
            reports_dir=str(r / "reports"),
        )

@dataclass(frozen=True)
class TrainConfig:
    top_n_series: int = 20

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