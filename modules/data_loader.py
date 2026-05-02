"""
data_loader.py
==============
Handles all data loading, cleaning, and preprocessing for the HHS 
Unaccompanied Children dashboard.
"""

import pandas as pd
import numpy as np
import os

# ─── Column name mapping for display ─────────────────────────────────────────
COLUMN_LABELS = {
    "Children_Apprehended":  "Children Apprehended",
    "Children_in_CBP":       "Children in CBP Custody",
    "Children_Transferred":  "Children Transferred",
    "Children_in_HHS":       "Children in HHS Care",
    "Children_Discharged":   "Children Discharged",
}

NUMERIC_COLS = list(COLUMN_LABELS.keys())


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV or Excel file.
    Supports .csv, .xlsx, and .xls formats.
    Returns a raw DataFrame before cleaning.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Please upload a CSV or Excel file.")

    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Cleans the raw dataframe:
      - Standardises column names
      - Parses and sorts dates
      - Coerces numeric columns
      - Fills or flags missing values
      - Removes duplicate rows

    Returns:
        df_clean  : cleaned DataFrame
        report    : dict with cleaning statistics for display
    """
    report = {}
    df = df.copy()

    # 1. Standardise column names (strip whitespace, title-case)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    report["original_rows"] = len(df)
    report["original_cols"] = len(df.columns)

    # 2. Parse the Date column
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        bad_dates = df[date_col].isna().sum()
        report["bad_dates_dropped"] = int(bad_dates)
        df = df.dropna(subset=[date_col])
        df = df.rename(columns={date_col: "Date"})
        df = df.sort_values("Date").reset_index(drop=True)
    else:
        report["bad_dates_dropped"] = 0

    # 3. Coerce numeric columns to numbers; count missing
    missing_before = 0
    missing_after = 0

    for col in df.columns:
        if col == "Date":
            continue
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after = df[col].isna().sum()
        missing_before += int(before)
        missing_after += int(after)

    report["missing_before_coerce"] = missing_before
    report["new_nulls_from_coerce"] = int(missing_after - missing_before)

    # 4. Fill remaining numeric nulls with column median
    nulls_filled = int(df[NUMERIC_COLS].isna().sum().sum()) if all(c in df.columns for c in NUMERIC_COLS) else 0
    numeric_in_df = [c for c in NUMERIC_COLS if c in df.columns]
    for col in numeric_in_df:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    report["nulls_filled_with_median"] = nulls_filled

    # 5. Remove exact duplicate rows
    dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(dupes)

    # 6. Derive helper columns
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["YearMonth"]  = df["Date"].dt.to_period("M")

    report["clean_rows"] = len(df)
    report["clean_cols"] = len(df.columns)

    return df, report


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a nicely formatted summary statistics table for numeric columns."""
    numeric_in_df = [c for c in NUMERIC_COLS if c in df.columns]
    stats = df[numeric_in_df].describe().T
    stats.index = [COLUMN_LABELS.get(i, i) for i in stats.index]
    stats.columns = ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"]
    return stats.round(1)


def filter_by_date(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Filters the dataframe to the given date range (inclusive)."""
    mask = (df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))
    return df.loc[mask].reset_index(drop=True)


def get_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates daily data to monthly averages/sums for trend analysis."""
    numeric_in_df = [c for c in NUMERIC_COLS if c in df.columns]
    monthly = (
        df.groupby(["Year", "Month", "Month_Name"])[numeric_in_df]
        .agg({
            "Children_Apprehended": "sum",
            "Children_in_CBP":      "mean",
            "Children_Transferred": "sum",
            "Children_in_HHS":      "mean",
            "Children_Discharged":  "sum",
        })
        .reset_index()
    )
    monthly = monthly.sort_values(["Year", "Month"]).reset_index(drop=True)
    monthly["Period"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)
    return monthly


def get_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates daily data to yearly totals/averages."""
    numeric_in_df = [c for c in NUMERIC_COLS if c in df.columns]
    yearly = (
        df.groupby("Year")[numeric_in_df]
        .agg({
            "Children_Apprehended": "sum",
            "Children_in_CBP":      "mean",
            "Children_Transferred": "sum",
            "Children_in_HHS":      "mean",
            "Children_Discharged":  "sum",
        })
        .reset_index()
    )
    return yearly.sort_values("Year").reset_index(drop=True)
