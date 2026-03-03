import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class Fold:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

def make_walk_forward_folds(series_df: pd.DataFrame, horizon_days: int, n_folds: int, step_days: int) -> list[Fold]:
    """
    Creates folds ending at:
      last_date - horizon
      last_date - horizon - step
      last_date - horizon - 2*step
      ...
    """
    last_date = series_df["date"].max()
    folds = []

    for k in range(n_folds):
        test_end = last_date - pd.Timedelta(days=k * step_days)
        test_start = test_end - pd.Timedelta(days=horizon_days - 1)
        train_end = test_start - pd.Timedelta(days=1)

        folds.append(Fold(train_end=train_end, test_start=test_start, test_end=test_end))

    # Sort folds chronologically (older to newer)
    folds = sorted(folds, key=lambda f: f.test_start)
    return folds