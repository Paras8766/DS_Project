"""
╔══════════════════════════════════════════════════════════════════╗
║  DSBDA Mini Project — Step 2: Exploratory Data Analysis (EDA)   ║
║  E-Commerce Sales Data Analysis & Future Sales Forecasting       ║
╚══════════════════════════════════════════════════════════════════╝

Plots generated (saved to outputs/plots/):
  1.  monthly_revenue.png            — Monthly revenue trend (line)
  2.  daily_revenue.png              — Daily revenue with 7-day rolling avg
  3.  top_categories.png             — Top 10 categories by revenue (bar)
  4.  top_states.png                 — Top 10 states by revenue (horizontal bar)
  5.  b2b_vs_b2c.png                 — B2B vs B2C revenue share (donut chart)
  6.  quarterly_revenue.png          — Revenue by quarter (bar)
  7.  dow_revenue.png                — Average revenue by day-of-week (bar)
  8.  amount_distribution.png        — Amount distribution (histogram + KDE)
  9.  category_heatmap.png           — Monthly revenue heatmap by category
  10. qty_vs_amount.png              — Qty vs Amount scatter coloured by category
  11. correlation_heatmap.png        — Feature correlation heatmap

Run:
    python eda.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "cleaned_sales.csv"
PLOTS_DIR  = BASE_DIR / "outputs" / "plots"

# ── Dark professional colour palette ─────────────────────────────
DARK_BG      = "#0D1117"
PANEL_BG     = "#161B22"
ACCENT       = "#58A6FF"       # electric blue
ACCENT2      = "#F78166"       # coral red
ACCENT3      = "#3FB950"       # emerald green
ACCENT4      = "#D2A8FF"       # soft violet
TEXT_PRIMARY = "#E6EDF3"
TEXT_MUTED   = "#8B949E"
GRID_COLOR   = "#21262D"

PALETTE = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#FFA657", "#79C0FF",
           "#56D364", "#FF7B72", "#D29922", "#A5D6FF"]

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def apply_dark_style() -> None:
    """Apply a consistent dark-theme to all matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   TEXT_PRIMARY,
        "axes.titlecolor":   TEXT_PRIMARY,
        "axes.titlesize":    13,
        "axes.labelsize":    10,
        "axes.titlepad":     14,
        "xtick.color":       TEXT_MUTED,
        "ytick.color":       TEXT_MUTED,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.6,
        "legend.facecolor":  PANEL_BG,
        "legend.edgecolor":  GRID_COLOR,
        "legend.labelcolor": TEXT_PRIMARY,
        "text.color":        TEXT_PRIMARY,
        "font.family":       "DejaVu Sans",
    })


def save(fig: plt.Figure, name: str) -> None:
    """Save figure and close it."""
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✔  {name}")


# ─────────────────────────────────────────────────────────────────
# Individual plot functions
# ─────────────────────────────────────────────────────────────────

def plot_monthly_revenue(df: pd.DataFrame) -> None:
    monthly = (
        df.groupby(df["Date"].dt.to_period("M"))["Amount"]
        .sum()
        .sort_index()
    )
    monthly.index = monthly.index.astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(monthly.index, monthly.values, alpha=0.18, color=ACCENT)
    ax.plot(monthly.index, monthly.values, color=ACCENT, linewidth=2.2, marker="o",
            markersize=5, markerfacecolor=DARK_BG, markeredgecolor=ACCENT, markeredgewidth=1.8)
    ax.set_title("Monthly Total Revenue")
    ax.set_xlabel("Month");  ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e3:.0f}K"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "monthly_revenue.png")


def plot_daily_revenue(df: pd.DataFrame) -> None:
    daily = df.groupby("Date")["Amount"].sum().reset_index()
    daily["rolling"] = daily["Amount"].rolling(7).mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(daily["Date"], daily["Amount"], color=ACCENT, alpha=0.35, width=0.8, label="Daily")
    ax.plot(daily["Date"], daily["rolling"], color=ACCENT2, linewidth=2, label="7-day avg")
    ax.set_title("Daily Revenue with 7-Day Rolling Average")
    ax.set_xlabel("Date");  ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e3:.0f}K"))
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "daily_revenue.png")


def plot_top_categories(df: pd.DataFrame) -> None:
    top = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(top.index.astype(str), top.values, color=PALETTE[:len(top)], edgecolor="none")
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top.max() * 0.01,
                f"₹{val/1e3:.0f}K", ha="center", va="bottom", fontsize=7.5, color=TEXT_MUTED)
    ax.set_title("Top 10 Categories by Revenue")
    ax.set_xlabel("Category");  ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e3:.0f}K"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "top_categories.png")


def plot_top_states(df: pd.DataFrame) -> None:
    top = (
        df.groupby("ship-state")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top.index.astype(str), top.values, color=PALETTE[:len(top)], edgecolor="none")
    for bar, val in zip(bars, top.values):
        ax.text(val + top.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"₹{val/1e3:.0f}K", va="center", fontsize=8, color=TEXT_MUTED)
    ax.set_title("Top 10 States by Revenue")
    ax.set_xlabel("Revenue (₹)");  ax.set_ylabel("State")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e3:.0f}K"))
    ax.grid(axis="x")
    fig.tight_layout()
    save(fig, "top_states.png")


def plot_b2b_vs_b2c(df: pd.DataFrame) -> None:
    seg = df.groupby("B2B")["Amount"].sum()

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        seg.values,
        labels=seg.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=[ACCENT, ACCENT2],
        wedgeprops={"width": 0.6, "edgecolor": DARK_BG, "linewidth": 2},
        pctdistance=0.75,
    )
    for t in texts:      t.set_color(TEXT_PRIMARY); t.set_fontsize(11)
    for t in autotexts:  t.set_color(DARK_BG);      t.set_fontsize(10); t.set_fontweight("bold")
    ax.set_title("B2B vs B2C Revenue Share")
    fig.tight_layout()
    save(fig, "b2b_vs_b2c.png")


def plot_quarterly_revenue(df: pd.DataFrame) -> None:
    df["YQ"] = df["Date"].dt.to_period("Q").astype(str)
    qtr = df.groupby("YQ")["Amount"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(qtr["YQ"], qtr["Amount"], color=ACCENT3, edgecolor="none")
    ax.set_title("Revenue by Quarter")
    ax.set_xlabel("Quarter");  ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e3:.0f}K"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "quarterly_revenue.png")


def plot_dow_revenue(df: pd.DataFrame) -> None:
    dow = df.groupby("DayOfWeek")["Amount"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [ACCENT2 if i >= 5 else ACCENT for i in range(7)]
    ax.bar([DAYS[i] for i in dow.index], dow.values, color=colors, edgecolor="none")
    ax.set_title("Average Order Revenue by Day of Week")
    ax.set_xlabel("Day");  ax.set_ylabel("Avg Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.grid(axis="y")
    # Legend for weekend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ACCENT, label="Weekday"),
                       Patch(facecolor=ACCENT2, label="Weekend")]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    save(fig, "dow_revenue.png")


def plot_amount_distribution(df: pd.DataFrame) -> None:
    amounts = df["Amount"].clip(upper=df["Amount"].quantile(0.99))

    fig, ax = plt.subplots(figsize=(9, 4))
    n, bins, patches = ax.hist(amounts, bins=60, color=ACCENT, edgecolor="none", alpha=0.7)

    # Simple KDE overlay using gaussian smoothing
    from scipy.stats import gaussian_kde  # noqa: E402 (stdlib fallback below)
    try:
        kde = gaussian_kde(amounts)
        x = np.linspace(amounts.min(), amounts.max(), 300)
        ax2 = ax.twinx()
        ax2.plot(x, kde(x), color=ACCENT2, linewidth=2)
        ax2.set_ylabel("Density", color=ACCENT2)
        ax2.tick_params(axis="y", labelcolor=ACCENT2)
        ax2.set_facecolor(PANEL_BG)
    except Exception:
        pass  # scipy not available — skip KDE

    ax.set_title("Order Amount Distribution")
    ax.set_xlabel("Amount (₹)");  ax.set_ylabel("Frequency")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "amount_distribution.png")


def plot_category_heatmap(df: pd.DataFrame) -> None:
    """Monthly revenue heatmap for top 8 categories."""
    top_cats = df.groupby("Category")["Amount"].sum().nlargest(8).index
    sub = df[df["Category"].isin(top_cats)].copy()
    sub["Month_str"] = sub["Date"].dt.to_period("M").astype(str)

    pivot = sub.pivot_table(index="Category", columns="Month_str", values="Amount",
                            aggfunc="sum", fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", origin="upper")
    ax.set_xticks(range(len(pivot.columns)));  ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)));    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Monthly Revenue Heatmap by Category")
    plt.colorbar(im, ax=ax, label="Revenue (₹)", fraction=0.03)
    fig.tight_layout()
    save(fig, "category_heatmap.png")


def plot_qty_vs_amount(df: pd.DataFrame) -> None:
    top_cats = df.groupby("Category")["Amount"].sum().nlargest(6).index
    sub = df[df["Category"].isin(top_cats)].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cat in enumerate(top_cats):
        mask = sub["Category"] == cat
        ax.scatter(sub.loc[mask, "Qty"], sub.loc[mask, "Amount"],
                   label=cat, alpha=0.45, s=18, color=PALETTE[i % len(PALETTE)])
    ax.set_title("Quantity vs Order Amount by Category")
    ax.set_xlabel("Quantity");  ax.set_ylabel("Amount (₹)")
    ax.legend(fontsize=7, markerscale=1.5)
    ax.grid()
    fig.tight_layout()
    save(fig, "qty_vs_amount.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_cols = ["Amount", "Qty", "Month", "DayOfWeek", "Week", "Quarter",
                "lag_1", "lag_7", "rolling_mean_7", "ewm_7"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols, fontsize=8)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if abs(corr.values[i, j]) > 0.5 else TEXT_MUTED)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    save(fig, "correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────
def main() -> None:
    print("═" * 60)
    print("  DSBDA Mini Project — Exploratory Data Analysis")
    print("═" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    apply_dark_style()

    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    print(f"\n  Loaded {len(df):,} rows from cleaned_sales.csv")
    print("\n  Generating plots:")

    plot_monthly_revenue(df)
    plot_daily_revenue(df)
    plot_top_categories(df)
    plot_top_states(df)
    plot_b2b_vs_b2c(df)
    plot_quarterly_revenue(df)
    plot_dow_revenue(df)
    plot_amount_distribution(df)
    plot_category_heatmap(df)
    plot_qty_vs_amount(df)
    plot_correlation_heatmap(df)

    print(f"\n  All plots saved → {PLOTS_DIR}\n")


if __name__ == "__main__":
    main()
