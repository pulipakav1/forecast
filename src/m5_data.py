import pandas as pd
import numpy as np

def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df

def load_raw_m5(paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(paths.sales_csv)
    calendar = pd.read_csv(paths.calendar_csv)
    prices = pd.read_csv(paths.prices_csv)
    return sales, calendar, prices

def melt_sales_to_long(sales: pd.DataFrame) -> pd.DataFrame:
    # Wide columns are d_1 ... d_1913 (or similar)
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

    # Merge calendar (contains date, wm_yr_wk, events, snap)
    df = long_df.merge(calendar, on="d", how="left")

    # Merge prices (store_id, item_id, wm_yr_wk -> sell_price)
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

    df["sell_price"] = df.groupby("id")["sell_price"].ffill().bfill()

    df = reduce_memory(df)
    return df

def select_top_series(df: pd.DataFrame, top_n: int) -> list[str]:
    totals = df.groupby("id")["sales"].sum().sort_values(ascending=False)
    return totals.head(top_n).index.tolist()