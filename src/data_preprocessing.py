"""
╔══════════════════════════════════════════════════════════════════╗
║  DSBDA Mini Project — Step 1: Data Preprocessing                ║
║  E-Commerce Sales Data Analysis & Future Sales Forecasting       ║
╚══════════════════════════════════════════════════════════════════╝

What this script does:
  1. Loads raw Amazon sales data
  2. Selects and validates required columns
  3. Filters only Shipped / Delivered orders
  4. Cleans and parses Date and Amount columns
  5. Engineers time-based features (Month, Week, DayOfWeek, Quarter, IsWeekend)
  6. Creates lag features and rolling statistics for ML models
  7. Detects and removes Amount outliers using the IQR method
  8. Saves the cleaned dataset to data/cleaned_sales.csv

Run:
    python data_preprocessing.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ── File paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE  = BASE_DIR / "data" / "amazon_sales.csv"
OUTPUT_FILE = BASE_DIR / "data" / "cleaned_sales.csv"

# ── Columns we need from the raw file ────────────────────────────
REQUIRED_COLUMNS = ["Date", "Status", "Category", "Qty", "Amount", "ship-state", "B2B"]

# ── IQR multiplier for outlier removal ───────────────────────────
IQR_MULTIPLIER = 3.0  # conservative — keeps most valid high-value orders


# ─────────────────────────────────────────────────────────────────
def load_raw_data(path: Path) -> pd.DataFrame:
    """Load CSV and keep only required columns."""
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in source file: {missing}")

    return df[REQUIRED_COLUMNS].copy()


def filter_valid_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Shipped or Delivered orders (exclude Cancelled, Pending, etc.)."""
    mask = df["Status"].astype(str).str.contains(
        "Shipped|Delivered", case=False, na=False, regex=True
    )
    filtered = df[mask].copy()
    print(f"  Orders after status filter : {len(filtered):,}  (removed {len(df) - len(filtered):,})")
    return filtered


def clean_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Date column; drop rows where parsing fails."""
    # Try MM-DD-YY first (Amazon export format), then let pandas infer
    df["Date"] = pd.to_datetime(df["Date"], format="%m-%d-%y", errors="coerce")
    df["Date"] = df["Date"].fillna(pd.to_datetime(df["Date"], errors="coerce"))
    before = len(df)
    df = df.dropna(subset=["Date"])
    print(f"  Rows after date cleaning   : {len(df):,}  (removed {before - len(df):,})")
    return df


def clean_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce Amount to numeric, drop nulls and zeros."""
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])
    df = df[df["Amount"] > 0].copy()
    return df


def remove_outliers_iqr(df: pd.DataFrame, col: str = "Amount") -> pd.DataFrame:
    """Remove extreme outliers using the IQR method."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - IQR_MULTIPLIER * IQR
    upper = Q3 + IQR_MULTIPLIER * IQR
    before = len(df)
    df = df[(df[col] >= lower) & (df[col] <= upper)].copy()
    print(f"  Rows after outlier removal : {len(df):,}  (removed {before - len(df):,})")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all time and lag-based features needed by ML models."""
    df = df.sort_values("Date").reset_index(drop=True)

    # ── Calendar features ─────────────────────────────────────────
    df["Month"]      = df["Date"].dt.month
    df["DayOfWeek"]  = df["Date"].dt.dayofweek          # 0 = Monday
    df["Week"]       = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]    = df["Date"].dt.quarter
    df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
    df["DayOfMonth"] = df["Date"].dt.day

    # ── Lag features (previous sales signals) ─────────────────────
    df["lag_1"]  = df["Amount"].shift(1)
    df["lag_7"]  = df["Amount"].shift(7)
    df["lag_30"] = df["Amount"].shift(30)

    # ── Rolling statistics ────────────────────────────────────────
    df["rolling_mean_7"]  = df["Amount"].rolling(7).mean()
    df["rolling_mean_30"] = df["Amount"].rolling(30).mean()
    df["rolling_std_7"]   = df["Amount"].rolling(7).std()

    # ── Exponentially weighted mean (trend-sensitive) ─────────────
    df["ewm_7"] = df["Amount"].ewm(span=7, adjust=False).mean()

    # Drop rows with NaN introduced by lag / rolling
    df = df.dropna(subset=["lag_1", "lag_7", "lag_30",
                            "rolling_mean_7", "rolling_mean_30",
                            "rolling_std_7", "ewm_7"]).copy()

    # ── Clean up string columns ───────────────────────────────────
    df["Category"]   = df["Category"].astype(str).str.strip().str.title()
    df["ship-state"] = df["ship-state"].astype(str).str.strip().str.upper()

    # Normalise B2B flag
    df["B2B"] = df["B2B"].astype(str).str.strip().str.lower().map(
        lambda x: "B2B" if x in {"true", "1", "yes", "y", "b2b"} else "B2C"
    )

    return df


def summarise(df: pd.DataFrame) -> None:
    """Print a brief summary of the cleaned dataset."""
    print("\n── Cleaned Dataset Summary ─────────────────────────────")
    print(f"  Shape      : {df.shape}")
    print(f"  Date range : {df['Date'].min().date()}  →  {df['Date'].max().date()}")
    print(f"  Categories : {df['Category'].nunique()}")
    print(f"  States     : {df['ship-state'].nunique()}")
    print(f"  Avg Amount : ₹{df['Amount'].mean():,.2f}")
    print(f"  Null count : {df.isnull().sum().sum()}")


def main() -> None:
    print("═" * 60)
    print("  DSBDA Mini Project — Data Preprocessing")
    print("═" * 60)

    df = load_raw_data(INPUT_FILE)
    print(f"\n  Raw rows loaded : {len(df):,}")

    df = filter_valid_orders(df)
    df = clean_date(df)
    df = clean_amount(df)
    df = remove_outliers_iqr(df)
    df = engineer_features(df)

    summarise(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  ✔  Cleaned data saved → {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
