"""
╔══════════════════════════════════════════════════════════════════╗
║  DSBDA Mini Project — Step 3: Predictive Modelling              ║
║  E-Commerce Sales Data Analysis & Future Sales Forecasting       ║
╚══════════════════════════════════════════════════════════════════╝

Models trained & compared:
  1. Linear Regression      — baseline, interpretable
  2. Decision Tree          — non-linear, rule-based splits
  3. Random Forest          — ensemble of trees, robust
  4. XGBoost Regressor      — gradient boosting, best accuracy

Install XGBoost once (if not already installed):
    pip install xgboost

Outputs saved to outputs/plots/:
  • actual_vs_predicted.png        — overlaid actual vs predicted line chart
  • residual_plots.png             — residual distribution for each model
  • feature_importance_rf.png      — Random Forest feature importances
  • feature_importance_xgb.png     — XGBoost feature importances
  • model_comparison_bar.png       — RMSE / MAE / R² bar chart comparison
  • forecast_next30.png            — 30-day future sales forecast (RF + XGB)

Run:
    python models.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# XGBoost — graceful fallback so the script still runs without it
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("  ⚠  XGBoost not installed. Run:  pip install xgboost")
    print("     XGBoost results will be skipped.\n")

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "cleaned_sales.csv"
PLOTS_DIR  = BASE_DIR / "outputs" / "plots"

# ── Modelling config ──────────────────────────────────────────────
FEATURE_COLUMNS = [
    "Month", "DayOfWeek", "Week", "Quarter", "IsWeekend", "DayOfMonth",
    "Qty",
    "lag_1", "lag_7", "lag_30",
    "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
    "ewm_7",
    "Category", "ship-state", "B2B",
]
TARGET_COLUMN = "Amount"
SPLIT_DATE    = pd.Timestamp("2022-06-01")   # chronological train/test split

# ── Dark colour palette ───────────────────────────────────────────
DARK_BG   = "#0D1117"
PANEL_BG  = "#161B22"
GRID_CLR  = "#21262D"
TEXT_CLR  = "#E6EDF3"
MUTED_CLR = "#8B949E"
MODEL_COLORS = {
    "Linear Regression": "#58A6FF",
    "Decision Tree":     "#F78166",
    "Random Forest":     "#3FB950",
    "XGBoost":           "#D2A8FF",
}


def apply_dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_CLR, "axes.labelcolor": TEXT_CLR,
        "axes.titlecolor": TEXT_CLR, "axes.titlesize": 13,
        "xtick.color": MUTED_CLR, "ytick.color": MUTED_CLR,
        "grid.color": GRID_CLR, "grid.linewidth": 0.6,
        "legend.facecolor": PANEL_BG, "legend.edgecolor": GRID_CLR,
        "legend.labelcolor": TEXT_CLR, "text.color": TEXT_CLR,
        "font.family": "DejaVu Sans",
    })


def save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✔  {name}")


# ─────────────────────────────────────────────────────────────────
# Data loading & preparation
# ─────────────────────────────────────────────────────────────────

def load_and_prepare(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load cleaned CSV, encode categoricals, and do a time-based split."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Keep only columns that exist (preprocessing version guard)
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]

    df_model = df[["Date"] + available_features + [TARGET_COLUMN]].copy()
    df_model = pd.get_dummies(df_model, columns=["Category", "ship-state", "B2B"],
                              dtype=int, drop_first=False)

    train = df_model[df_model["Date"] < SPLIT_DATE].copy()
    test  = df_model[df_model["Date"] >= SPLIT_DATE].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"Train or test set is empty. Adjust SPLIT_DATE ({SPLIT_DATE.date()})."
        )

    feature_cols = [c for c in train.columns if c not in ["Date", TARGET_COLUMN]]
    X_train = train[feature_cols]
    y_train = train[TARGET_COLUMN]
    X_test  = test[feature_cols]
    y_test  = test[TARGET_COLUMN]

    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test, feature_cols


# ─────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE%": mape}


# ─────────────────────────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────────────────────────

def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds, compute_metrics(y_test, preds)


def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds, compute_metrics(y_test, preds)


def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds, compute_metrics(y_test, preds)


def train_xgboost(X_train, X_test, y_train, y_test):
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    preds = model.predict(X_test)
    return model, preds, compute_metrics(y_test, preds)


# ─────────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_test: pd.Series, results: dict) -> None:
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 3.5 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]

    x = np.arange(len(y_test))
    for ax, (name, (_, preds, _)) in zip(axes, results.items()):
        ax.plot(x, y_test.values, color="#E6EDF3", linewidth=0.8, alpha=0.7, label="Actual")
        ax.plot(x, preds, color=MODEL_COLORS.get(name, "#58A6FF"),
                linewidth=1.2, alpha=0.9, label=name)
        ax.set_ylabel("Amount (₹)", fontsize=8)
        ax.set_title(f"{name} — Actual vs Predicted", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
        ax.legend(fontsize=7)
        ax.grid(axis="y")

    axes[-1].set_xlabel("Test Sample Index")
    fig.tight_layout()
    save(fig, "actual_vs_predicted.png")


def plot_residuals(y_test: pd.Series, results: dict) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, (_, preds, _)) in zip(axes, results.items()):
        residuals = y_test.values - preds
        ax.hist(residuals, bins=40, color=MODEL_COLORS.get(name, "#58A6FF"),
                alpha=0.8, edgecolor="none")
        ax.axvline(0, color=MUTED_CLR, linewidth=1.2, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Count")
        ax.grid(axis="y")

    fig.suptitle("Residual Distributions", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "residual_plots.png")


def plot_feature_importance(model, feature_names: list[str], model_name: str, filename: str) -> None:
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    else:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    color = MODEL_COLORS.get(model_name, "#58A6FF")
    ax.bar(fi.index, fi.values, color=color, edgecolor="none")
    ax.set_title(f"{model_name} — Feature Importance (Top 15)")
    ax.set_xlabel("Feature");  ax.set_ylabel("Importance")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, filename)


def plot_model_comparison(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["RMSE", "MAE", "R²"]
    bar_colors = [MODEL_COLORS.get(m, "#58A6FF") for m in metrics_df["Model"]]

    for ax, metric in zip(axes, metrics):
        vals = metrics_df[metric].values
        bars = ax.bar(metrics_df["Model"], vals, color=bar_colors, edgecolor="none")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=MUTED_CLR)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y")

    fig.suptitle("Model Performance Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "model_comparison_bar.png")


def plot_forecast_next30(df_original: pd.DataFrame, rf_model, xgb_model,
                         feature_cols: list[str]) -> None:
    """Recursive 30-day ahead forecast using the best two models."""
    history = df_original.copy().sort_values("Date").reset_index(drop=True)
    last_date = history["Date"].max()

    forecasts_rf  = []
    forecasts_xgb = []

    for i in range(1, 31):
        next_date = last_date + pd.Timedelta(days=i)

        row = {
            "Month":            float(next_date.month),
            "DayOfWeek":        float(next_date.dayofweek),
            "Week":             float(next_date.isocalendar().week),
            "Quarter":          float(next_date.quarter),
            "IsWeekend":        float(int(next_date.dayofweek >= 5)),
            "DayOfMonth":       float(next_date.day),
            "Qty":              float(history["Qty"].tail(7).mean()),
            "lag_1":            float(history["Amount"].iloc[-1]),
            "lag_7":            float(history["Amount"].iloc[-7] if len(history) >= 7 else history["Amount"].mean()),
            "lag_30":           float(history["Amount"].iloc[-30] if len(history) >= 30 else history["Amount"].mean()),
            "rolling_mean_7":   float(history["Amount"].tail(7).mean()),
            "rolling_mean_30":  float(history["Amount"].tail(30).mean()),
            "rolling_std_7":    float(history["Amount"].tail(7).std()),
            "ewm_7":            float(history["Amount"].ewm(span=7).mean().iloc[-1]),
            "Category":         history.groupby("Category")["Amount"].sum().idxmax(),
            "ship-state":       history.groupby("ship-state")["Amount"].sum().idxmax(),
            "B2B":              "B2C",
        }

        row_df = pd.DataFrame([row])
        row_enc = pd.get_dummies(row_df, columns=["Category", "ship-state", "B2B"],
                                 dtype=int, drop_first=False)
        row_enc = row_enc.reindex(columns=feature_cols, fill_value=0)

        pred_rf  = float(rf_model.predict(row_enc)[0])
        forecasts_rf.append(pred_rf)

        if xgb_model is not None:
            pred_xgb = float(xgb_model.predict(row_enc)[0])
            forecasts_xgb.append(pred_xgb)

        # Append RF prediction to history for next iteration (recursive)
        new_row = history.iloc[-1].copy()
        new_row["Date"]   = next_date
        new_row["Amount"] = pred_rf
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]

    # Plot
    fig, ax = plt.subplots(figsize=(13, 5))
    # Historical tail
    hist_tail = df_original.sort_values("Date").tail(60)
    ax.plot(hist_tail["Date"], hist_tail["Amount"], color="#E6EDF3",
            linewidth=1, alpha=0.5, label="Historical")
    ax.plot(future_dates, forecasts_rf, color=MODEL_COLORS["Random Forest"],
            linewidth=2, marker="o", markersize=4, label="RF Forecast")
    if forecasts_xgb:
        ax.plot(future_dates, forecasts_xgb, color=MODEL_COLORS["XGBoost"],
                linewidth=2, marker="s", markersize=4, linestyle="--", label="XGB Forecast")
    ax.axvline(last_date, color=MUTED_CLR, linewidth=1, linestyle=":", label="Forecast start")
    ax.set_title("30-Day Future Sales Forecast")
    ax.set_xlabel("Date");  ax.set_ylabel("Predicted Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "forecast_next30.png")


# ─────────────────────────────────────────────────────────────────
def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    print("\n" + "═" * 62)
    print(f"  {'Model':<22} {'RMSE':>10} {'MAE':>10} {'R²':>8} {'MAPE%':>8}")
    print("─" * 62)
    for _, row in metrics_df.iterrows():
        print(f"  {row['Model']:<22} {row['RMSE']:>10,.2f} {row['MAE']:>10,.2f} "
              f"{row['R²']:>8.4f} {row['MAPE%']:>7.2f}%")
    print("═" * 62)


# ─────────────────────────────────────────────────────────────────
def main() -> None:
    print("═" * 60)
    print("  DSBDA Mini Project — Predictive Modelling")
    print("═" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    apply_dark_style()

    # ── Prepare data ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_cols = load_and_prepare(INPUT_FILE)

    # Load original df for forecast plot
    df_orig = pd.read_csv(INPUT_FILE)
    df_orig["Date"] = pd.to_datetime(df_orig["Date"], errors="coerce")
    df_orig = df_orig.dropna(subset=["Date"])

    results = {}   # name → (model, preds, metrics)

    # ── 1. Linear Regression ──────────────────────────────────────
    print("\n  [1/4] Linear Regression …", end=" ", flush=True)
    lr_model, lr_preds, lr_metrics = train_linear_regression(X_train, X_test, y_train, y_test)
    results["Linear Regression"] = (lr_model, lr_preds, lr_metrics)
    print("done")

    # ── 2. Decision Tree ──────────────────────────────────────────
    print("  [2/4] Decision Tree …", end=" ", flush=True)
    dt_model, dt_preds, dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test)
    results["Decision Tree"] = (dt_model, dt_preds, dt_metrics)
    print("done")

    # ── 3. Random Forest ─────────────────────────────────────────
    print("  [3/4] Random Forest …", end=" ", flush=True)
    rf_model, rf_preds, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    results["Random Forest"] = (rf_model, rf_preds, rf_metrics)
    print("done")

    # ── 4. XGBoost ────────────────────────────────────────────────
    xgb_model = None
    if XGBOOST_AVAILABLE:
        print("  [4/4] XGBoost …", end=" ", flush=True)
        xgb_model, xgb_preds, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
        results["XGBoost"] = (xgb_model, xgb_preds, xgb_metrics)
        print("done")
    else:
        print("  [4/4] XGBoost — skipped (not installed)")

    # ── Metrics table ─────────────────────────────────────────────
    rows = [{"Model": name, **m} for name, (_, _, m) in results.items()]
    metrics_df = pd.DataFrame(rows)
    print_metrics_table(metrics_df)

    # ── Plots ─────────────────────────────────────────────────────
    print("\n  Saving plots:")
    plot_actual_vs_predicted(y_test, results)
    plot_residuals(y_test, results)
    plot_feature_importance(rf_model, feature_cols, "Random Forest", "feature_importance_rf.png")
    if xgb_model is not None:
        plot_feature_importance(xgb_model, feature_cols, "XGBoost", "feature_importance_xgb.png")
    plot_model_comparison(metrics_df)
    plot_forecast_next30(df_orig, rf_model, xgb_model, feature_cols)

    print(f"\n  All outputs saved → {PLOTS_DIR}\n")


if __name__ == "__main__":
    main()
