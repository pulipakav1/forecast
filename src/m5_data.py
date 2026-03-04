import pandas as pd
import numpy as np
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str) -> str:
    """Resolve path: prefer project-root-relative so notebooks/CLI both work."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    under_root = _PROJECT_ROOT.joinpath(*Path(path).parts)
    if under_root.exists():
        return str(under_root.resolve())
    cwd_path = p.resolve()
    if cwd_path.exists():
        return str(cwd_path)
    return str(p)


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df

def load_raw_m5(paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _read(path_key: str, path_value: str) -> pd.DataFrame:
        resolved = _resolve(path_value)
        if not Path(resolved).exists():
            raise FileNotFoundError(
                f"Data file not found: {path_value}\n"
                f"  Resolved to: {Path(resolved).absolute()}\n"
                f"  Project root used: {_PROJECT_ROOT}\n"
                f"  Put M5 CSV files under: {_PROJECT_ROOT / 'data' / 'm5'}"
            )
        return pd.read_csv(resolved)

    sales = _read("sales_csv", paths.sales_csv)
    calendar = _read("calendar_csv", paths.calendar_csv)
    prices = _read("prices_csv", paths.prices_csv)
    return sales, calendar, prices

def melt_sales_to_long(sales: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols = [c for c in sales.columns if c.startswith("d_")]

    long_df = sales.melt(
        id_vars=id_cols,
        value_vars=d_cols,
        var_name="d",
        value_name="sales"
    )
    return long_df

def build_merged_long(paths) -> pd.DataFrame:
    sales, calendar, prices = load_raw_m5(paths)
    long_df = melt_sales_to_long(sales)

    df = long_df.merge(calendar, on="d", how="left")

    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    keep = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "date", "sales", "sell_price",
        "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI"
    ]
    df = df[keep]

    df = df.sort_values(["id", "date"]).reset_index(drop=True)


    df["sell_price"] = df.groupby("id")["sell_price"].transform(lambda s: s.ffill().bfill())
    df["sell_price"] = df["sell_price"].fillna(0.0)

    df = reduce_memory(df)
    return df

def load_one_series(paths, series_id: str) -> pd.DataFrame:
    """
    Load M5 data for a single series only (low memory; for notebooks / single-series scripts).
    Returns same schema as build_merged_long but with one id.
    """
    calendar_path = _resolve(paths.calendar_csv)
    prices_path = _resolve(paths.prices_csv)
    sales_path = _resolve(paths.sales_csv)
    if not Path(sales_path).exists():
        raise FileNotFoundError(f"Data not found: {sales_path}")

    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_one = None
    for chunk in pd.read_csv(sales_path, chunksize=5000, low_memory=False):
        match = chunk[chunk["id"] == series_id]
        if len(match) > 0:
            sales_one = match.iloc[:1]
            break
    if sales_one is None:
        raise ValueError(f"series_id not found in sales: {series_id}")

    d_cols = [c for c in sales_one.columns if c.startswith("d_")]
    long_one = sales_one.melt(
        id_vars=id_cols,
        value_vars=d_cols,
        var_name="d",
        value_name="sales",
    )
    df = long_one.merge(calendar, on="d", how="left")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df["date"] = pd.to_datetime(df["date"])
    keep = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "date", "sales", "sell_price",
        "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    df = df[[c for c in keep if c in df.columns]]
    df = df.sort_values("date").reset_index(drop=True)
    df["sell_price"] = df["sell_price"].ffill().bfill().fillna(0.0)
    df = reduce_memory(df)
    return df


def select_top_series(df: pd.DataFrame, top_n: int) -> list[str]:
    totals = df.groupby("id")["sales"].sum().sort_values(ascending=False)
    return totals.head(top_n).index.tolist()