import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class Fold:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

def make_walk_forward_folds(
    series_df: pd.DataFrame,
    horizon_days: int,
    n_folds: int,
    step_days: int,
    min_train_days: int = 180
) -> list[Fold]:

    series_df = series_df.sort_values("date").reset_index(drop=True)
    dates = series_df["date"].unique()
    dates = pd.DatetimeIndex(dates).sort_values()
    first_date = dates.min()
    last_date = dates.max()

    folds = []

    for k in range(n_folds):
        end_idx = len(dates) - 1 - k * step_days
        start_idx = end_idx - horizon_days + 1

        if start_idx < 0:
            continue

        test_start = dates[start_idx]
        test_end = dates[end_idx]
        train_end_idx = start_idx - 1
        if train_end_idx < 0:
            continue
        train_end = dates[train_end_idx]

        if test_start < first_date:
            continue

        train_days = (train_end - first_date).days
        if train_days < min_train_days:
            continue

        folds.append(
            Fold(
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    folds = sorted(folds, key=lambda f: f.test_start)

    print("Generated folds:", len(folds))
    return folds