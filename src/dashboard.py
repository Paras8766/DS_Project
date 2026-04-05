"""
╔══════════════════════════════════════════════════════════════════╗
║  DSBDA Mini Project — Step 4: Streamlit Dashboard               ║
║  E-Commerce Sales Data Analysis & Future Sales Forecasting       ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    streamlit run dashboard.py

Features:
  • Sidebar: Category, State, B2B/B2C, Date range filters
  • KPI row: Revenue, Orders, AOV, Predicted next-day revenue
  • Monthly & daily revenue charts (Plotly dark theme)
  • Category and State revenue bar charts
  • Heatmap: monthly revenue by category
  • Interactive prediction panel (Qty, Category, State → predicted revenue)
  • Model metrics comparison table (LR, DT, RF, XGBoost)
  • 30-day forecast chart

Install dependencies (once):
    pip install streamlit plotly scikit-learn xgboost pandas numpy
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "cleaned_sales.csv"

# ── Features & target ────────────────────────────────────────────
FEATURE_COLUMNS = [
    "Month", "DayOfWeek", "Week", "Quarter", "IsWeekend", "DayOfMonth",
    "Qty",
    "lag_1", "lag_7", "lag_30",
    "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
    "ewm_7",
    "Category", "ship-state", "B2B",
]
TARGET  = "Amount"
SPLIT_DATE = pd.Timestamp("2022-06-01")

# ── Plotly dark template ──────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT          = "#58A6FF"
ACCENT2         = "#F78166"
ACCENT3         = "#3FB950"
ACCENT4         = "#D2A8FF"
MODEL_COLORS    = {
    "Linear Regression": ACCENT,
    "Decision Tree":     ACCENT2,
    "Random Forest":     ACCENT3,
    "XGBoost":           ACCENT4,
}


# ═══════════════════════════════════════════════════════════════
# Data & Model helpers  (cached)
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df


def _encode(df: pd.DataFrame, existing_cols: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode categorical columns and optionally align to a column list."""
    cat_cols = [c for c in ["Category", "ship-state", "B2B"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, dtype=int, drop_first=False)
    if existing_cols is not None:
        df = df.reindex(columns=existing_cols, fill_value=0)
    return df


@st.cache_resource
def train_all_models() -> dict:
    """Train LR, DT, RF (and XGBoost if available) on the full cleaned dataset.
    Returns a dict with model objects, feature columns, defaults, and test-set metrics."""
    df = load_data().sort_values("Date").reset_index(drop=True)

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    df_model = df[["Date"] + available_features + [TARGET]].copy()
    df_model = _encode(df_model)

    train_df = df_model[df_model["Date"] < SPLIT_DATE].copy()
    test_df  = df_model[df_model["Date"] >= SPLIT_DATE].copy()

    feat_cols = [c for c in train_df.columns if c not in ["Date", TARGET]]
    X_train, y_train = train_df[feat_cols], train_df[TARGET]
    X_test,  y_test  = test_df[feat_cols],  test_df[TARGET]

    def _metrics(y_true, y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
        return {"RMSE": round(rmse, 2), "MAE": round(mae, 2),
                "R²": round(r2, 4), "MAPE%": round(mape, 2)}

    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = {"model": lr, "metrics": _metrics(y_test, lr.predict(X_test))}

    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)
    dt.fit(X_train, y_train)
    models["Decision Tree"] = {"model": dt, "metrics": _metrics(y_test, dt.predict(X_test))}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                               min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = {"model": rf, "metrics": _metrics(y_test, rf.predict(X_test))}

    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbosity=0)
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        models["XGBoost"] = {"model": xgb, "metrics": _metrics(y_test, xgb.predict(X_test))}

    # Default values for interactive prediction
    last_date = df["Date"].max() + pd.Timedelta(days=1)
    defaults = {
        "Month":           float(last_date.month),
        "DayOfWeek":       float(last_date.dayofweek),
        "Week":            float(last_date.isocalendar().week),
        "Quarter":         float(last_date.quarter),
        "IsWeekend":       float(int(last_date.dayofweek >= 5)),
        "DayOfMonth":      float(last_date.day),
        "lag_1":           float(df["Amount"].iloc[-1]),
        "lag_7":           float(df["Amount"].iloc[-7] if len(df) >= 7 else df["Amount"].mean()),
        "lag_30":          float(df["Amount"].iloc[-30] if len(df) >= 30 else df["Amount"].mean()),
        "rolling_mean_7":  float(df["Amount"].tail(7).mean()),
        "rolling_mean_30": float(df["Amount"].tail(30).mean()),
        "rolling_std_7":   float(df["Amount"].tail(7).std()),
        "ewm_7":           float(df["Amount"].ewm(span=7).mean().iloc[-1]),
    }

    return {"models": models, "feat_cols": feat_cols, "defaults": defaults, "df": df}


def predict_single(row_dict: dict, feat_cols: list[str], model) -> float:
    """Encode a single feature-row dict and return a model prediction."""
    row_df  = pd.DataFrame([row_dict])
    row_enc = _encode(row_df, existing_cols=feat_cols)
    return float(model.predict(row_enc)[0])


def build_forecast(df: pd.DataFrame, model, feat_cols: list[str],
                   days: int = 30) -> pd.DataFrame:
    """Recursive multi-step forecast for `days` future days."""
    history = df.copy().sort_values("Date").reset_index(drop=True)
    last_date = history["Date"].max()
    preds = []

    for i in range(1, days + 1):
        nd = last_date + pd.Timedelta(days=i)
        row = {
            "Month": float(nd.month), "DayOfWeek": float(nd.dayofweek),
            "Week":  float(nd.isocalendar().week), "Quarter": float(nd.quarter),
            "IsWeekend": float(int(nd.dayofweek >= 5)), "DayOfMonth": float(nd.day),
            "Qty":  float(history["Qty"].tail(7).mean()),
            "lag_1":  float(history["Amount"].iloc[-1]),
            "lag_7":  float(history["Amount"].iloc[-7] if len(history) >= 7 else history["Amount"].mean()),
            "lag_30": float(history["Amount"].iloc[-30] if len(history) >= 30 else history["Amount"].mean()),
            "rolling_mean_7":  float(history["Amount"].tail(7).mean()),
            "rolling_mean_30": float(history["Amount"].tail(30).mean()),
            "rolling_std_7":   float(history["Amount"].tail(7).std()),
            "ewm_7":           float(history["Amount"].ewm(span=7).mean().iloc[-1]),
            "Category":   history.groupby("Category")["Amount"].sum().idxmax(),
            "ship-state": history.groupby("ship-state")["Amount"].sum().idxmax(),
            "B2B": "B2C",
        }
        pred = predict_single(row, feat_cols, model)
        preds.append({"Date": nd, "Forecast": pred})
        new_row = history.iloc[-1].copy()
        new_row["Date"] = nd;  new_row["Amount"] = pred
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(preds)


# ═══════════════════════════════════════════════════════════════
# Page layout helpers
# ═══════════════════════════════════════════════════════════════

def kpi_card(col, label: str, value: str, delta: str | None = None) -> None:
    col.metric(label, value, delta)


def section(title: str) -> None:
    st.markdown(f"""
    <div style="margin-top:2rem;margin-bottom:0.3rem;
                border-left:3px solid {ACCENT};padding-left:10px;">
        <span style="font-size:1.05rem;font-weight:600;color:{ACCENT};">{title}</span>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Main dashboard
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="E-Commerce Sales Forecasting",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global CSS override (dark professional) ───────────────────
    st.markdown("""
    <style>
    /* Overall background */
    .stApp { background-color: #0D1117; }
    section[data-testid="stSidebar"] { background-color: #161B22; }
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label { color: #8B949E !important; font-size:0.78rem; }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #E6EDF3 !important; font-size: 1.55rem; font-weight: 700;
    }
    /* Table */
    thead tr th { background: #161B22 !important; color: #58A6FF !important; }
    tbody tr:nth-child(even) { background: #161B22; }
    /* Sidebar text */
    .css-1d391kg, [data-testid="stSidebar"] * { color: #E6EDF3; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:1.5rem;">
        <div style="font-size:2.4rem;">📊</div>
        <div>
            <h1 style="margin:0;color:#E6EDF3;font-size:1.7rem;font-weight:700;">
                E-Commerce Sales Forecasting Dashboard
            </h1>
            <p style="margin:2px 0 0;color:#8B949E;font-size:0.85rem;">
                DSBDA Mini Project · Amazon Sales Dataset · Models: LR · DT · RF · XGBoost
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data & models ────────────────────────────────────────
    df_full = load_data()
    cache   = train_all_models()
    all_models  = cache["models"]
    feat_cols   = cache["feat_cols"]
    defaults    = cache["defaults"]

    # ── Sidebar filters ───────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 🔍 Filters")

        cats = sorted(df_full["Category"].dropna().unique())
        sel_cats = st.multiselect("Category", cats, default=cats)

        states = sorted(df_full["ship-state"].dropna().unique())
        sel_states = st.multiselect("State", states, default=states)

        b2b_opts = sorted(df_full["B2B"].dropna().unique()) if "B2B" in df_full.columns else []
        sel_b2b  = st.multiselect("Segment", b2b_opts, default=b2b_opts) if b2b_opts else b2b_opts

        min_d = df_full["Date"].min().date()
        max_d = df_full["Date"].max().date()
        date_range = st.date_input("Date Range", (min_d, max_d), min_value=min_d, max_value=max_d)
        start_d, end_d = (date_range if isinstance(date_range, tuple) and len(date_range) == 2
                          else (min_d, max_d))

        st.markdown("---")
        sel_model_name = st.selectbox("Forecast model", list(all_models.keys()))
        forecast_days  = st.slider("Forecast horizon (days)", 7, 60, 30, 1)

    # ── Apply filters ─────────────────────────────────────────────
    fdf = df_full.copy()
    if sel_cats:   fdf = fdf[fdf["Category"].isin(sel_cats)]
    if sel_states: fdf = fdf[fdf["ship-state"].isin(sel_states)]
    if sel_b2b and "B2B" in fdf.columns:
        fdf = fdf[fdf["B2B"].isin(sel_b2b)]
    fdf = fdf[(fdf["Date"].dt.date >= start_d) & (fdf["Date"].dt.date <= end_d)]

    if fdf.empty:
        st.warning("⚠️  No data available for the selected filters. Adjust the sidebar.")
        return

    # ── KPIs ─────────────────────────────────────────────────────
    section("📈 Key Performance Indicators")
    total_rev    = fdf["Amount"].sum()
    total_orders = len(fdf)
    aov          = total_rev / total_orders if total_orders else 0
    top_cat      = fdf.groupby("Category")["Amount"].sum().idxmax()
    top_state    = fdf.groupby("ship-state")["Amount"].sum().idxmax()

    # Next-day prediction using selected model
    sel_model = all_models[sel_model_name]["model"]
    next_day_row = {**defaults,
                    "Qty":        float(fdf["Qty"].tail(7).mean()),
                    "Category":   fdf.groupby("Category")["Amount"].sum().idxmax(),
                    "ship-state": fdf.groupby("ship-state")["Amount"].sum().idxmax(),
                    "B2B":        "B2C"}
    next_day_pred = predict_single(next_day_row, feat_cols, sel_model)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi_card(k1, "💰 Total Revenue",      f"₹{total_rev:,.0f}")
    kpi_card(k2, "🛒 Total Orders",       f"{total_orders:,}")
    kpi_card(k3, "📦 Avg Order Value",    f"₹{aov:,.2f}")
    kpi_card(k4, "🏆 Top Category",       str(top_cat))
    kpi_card(k5, "📍 Top State",          str(top_state))
    kpi_card(k6, "🔮 Next-Day Forecast",  f"₹{next_day_pred:,.0f}")

    # ── Monthly revenue line chart ────────────────────────────────
    section("📅 Revenue Over Time")
    monthly = (fdf.groupby(fdf["Date"].dt.to_period("M"))["Amount"]
               .sum().reset_index())
    monthly["Date"] = monthly["Date"].astype(str)

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly["Amount"],
        mode="lines+markers", name="Monthly Revenue",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=6, color=ACCENT),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"
    ))
    fig_monthly.update_layout(template=PLOTLY_TEMPLATE, title="Monthly Total Revenue",
                              xaxis_title="Month", yaxis_title="Revenue (₹)",
                              height=340, margin=dict(t=50, b=40))
    st.plotly_chart(fig_monthly, use_container_width=True)

    # ── Daily + rolling ──────────────────────────────────────────
    daily = fdf.groupby("Date")["Amount"].sum().reset_index()
    daily["rolling_7"] = daily["Amount"].rolling(7).mean()

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(x=daily["Date"], y=daily["Amount"],
                               name="Daily", marker_color=ACCENT, opacity=0.35))
    fig_daily.add_trace(go.Scatter(x=daily["Date"], y=daily["rolling_7"],
                                   mode="lines", name="7-day avg",
                                   line=dict(color=ACCENT2, width=2)))
    fig_daily.update_layout(template=PLOTLY_TEMPLATE, title="Daily Revenue with 7-Day Rolling Average",
                             height=300, margin=dict(t=50, b=40))
    st.plotly_chart(fig_daily, use_container_width=True)

    # ── Category and State bar charts ─────────────────────────────
    section("🗂️  Revenue Breakdown")
    col_a, col_b = st.columns(2)

    cat_rev = (fdf.groupby("Category")["Amount"].sum()
               .sort_values(ascending=False).head(10).reset_index())
    fig_cat = px.bar(cat_rev, x="Category", y="Amount",
                     title="Top 10 Categories by Revenue",
                     color="Amount", color_continuous_scale="Blues",
                     template=PLOTLY_TEMPLATE)
    fig_cat.update_layout(height=360, coloraxis_showscale=False)
    col_a.plotly_chart(fig_cat, use_container_width=True)

    state_rev = (fdf.groupby("ship-state")["Amount"].sum()
                 .sort_values(ascending=False).head(10).reset_index())
    fig_state = px.bar(state_rev, x="Amount", y="ship-state",
                       orientation="h", title="Top 10 States by Revenue",
                       color="Amount", color_continuous_scale="Teal",
                       template=PLOTLY_TEMPLATE)
    fig_state.update_layout(height=360, coloraxis_showscale=False,
                             yaxis=dict(autorange="reversed"))
    col_b.plotly_chart(fig_state, use_container_width=True)

    # ── B2B vs B2C donut ─────────────────────────────────────────
    if "B2B" in fdf.columns:
        seg = fdf.groupby("B2B")["Amount"].sum().reset_index()
        col_c, col_d = st.columns([1, 2])
        fig_b2b = px.pie(seg, names="B2B", values="Amount",
                         title="B2B vs B2C Revenue", hole=0.55,
                         color_discrete_sequence=[ACCENT, ACCENT2],
                         template=PLOTLY_TEMPLATE)
        fig_b2b.update_layout(height=300)
        col_c.plotly_chart(fig_b2b, use_container_width=True)

        # Quarterly revenue
        fdf2 = fdf.copy()
        fdf2["Quarter"] = fdf2["Date"].dt.to_period("Q").astype(str)
        qtr = fdf2.groupby("Quarter")["Amount"].sum().reset_index()
        fig_qtr = px.bar(qtr, x="Quarter", y="Amount",
                         title="Revenue by Quarter", color="Amount",
                         color_continuous_scale="Greens", template=PLOTLY_TEMPLATE)
        fig_qtr.update_layout(height=300, coloraxis_showscale=False)
        col_d.plotly_chart(fig_qtr, use_container_width=True)

    # ── Category monthly heatmap ──────────────────────────────────
    section("🔥 Category × Month Heatmap")
    top_cats_hm = fdf.groupby("Category")["Amount"].sum().nlargest(8).index
    sub_hm = fdf[fdf["Category"].isin(top_cats_hm)].copy()
    sub_hm["Month_str"] = sub_hm["Date"].dt.to_period("M").astype(str)
    pivot = sub_hm.pivot_table(index="Category", columns="Month_str",
                                values="Amount", aggfunc="sum", fill_value=0)
    fig_heat = px.imshow(pivot, color_continuous_scale="Blues",
                         title="Monthly Revenue by Category",
                         template=PLOTLY_TEMPLATE, aspect="auto")
    fig_heat.update_layout(height=340)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Interactive prediction ────────────────────────────────────
    section("🧮 Interactive Revenue Prediction")
    p1, p2, p3, p4 = st.columns(4)
    qty      = p1.number_input("Quantity", min_value=1, value=5, step=1)
    category = p2.selectbox("Category", sorted(df_full["Category"].dropna().unique()))
    state    = p3.selectbox("State",    sorted(df_full["ship-state"].dropna().unique()))
    b2b_seg  = p4.selectbox("Segment",  ["B2C", "B2B"])
    pred_model_name = st.selectbox("Model for prediction", list(all_models.keys()), key="pred_model")

    pred_row = {
        **defaults,
        "Qty": float(qty), "Category": category,
        "ship-state": state, "B2B": b2b_seg,
    }
    pred_val = predict_single(pred_row, feat_cols, all_models[pred_model_name]["model"])
    st.markdown(f"""
    <div style="background:#161B22;border:1px solid #21262D;border-radius:10px;
                padding:20px 28px;display:inline-block;margin-top:8px;">
        <span style="color:#8B949E;font-size:0.8rem;">Predicted Revenue ({pred_model_name})</span><br>
        <span style="color:{ACCENT};font-size:2rem;font-weight:700;">₹{pred_val:,.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── 30-day Forecast ───────────────────────────────────────────
    section(f"🔭 {forecast_days}-Day Sales Forecast  ({sel_model_name})")
    with st.spinner("Generating forecast …"):
        forecast_df = build_forecast(fdf, sel_model, feat_cols, days=forecast_days)

    hist_tail = fdf.groupby("Date")["Amount"].sum().reset_index().tail(60)
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=hist_tail["Date"], y=hist_tail["Amount"],
                                mode="lines", name="Historical",
                                line=dict(color="#E6EDF3", width=1.2), opacity=0.5))
    fig_fc.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"],
                                mode="lines+markers", name=f"{sel_model_name} Forecast",
                                line=dict(color=MODEL_COLORS.get(sel_model_name, ACCENT),
                                          width=2.2, dash="dot"),
                                marker=dict(size=5)))
    today_marker = pd.Timestamp(fdf["Date"].max())
    fig_fc.add_vline(x=today_marker, line_dash="dot", line_color="#8B949E")
    fig_fc.add_annotation(
        x=today_marker,
        y=1,
        yref="paper",
        text="Today",
        showarrow=False,
        yshift=10,
        font=dict(color="#8B949E", size=11),
    )
    fig_fc.update_layout(template=PLOTLY_TEMPLATE,
                          title=f"{forecast_days}-Day Revenue Forecast",
                          xaxis_title="Date", yaxis_title="Revenue (₹)",
                          height=380, margin=dict(t=50, b=40))
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Model metrics table ───────────────────────────────────────
    section("📊 Model Performance Comparison")
    if not XGBOOST_AVAILABLE:
        st.info("💡 XGBoost not installed — run `pip install xgboost` to include it.")

    rows = [{"Model": name, **info["metrics"]} for name, info in all_models.items()]
    metrics_df = pd.DataFrame(rows)

    # Highlight best value per metric
    def highlight_best(col):
        if col.name in ("R²",):          # higher is better
            best = col.max()
            return ["background-color:#1c3a2a;color:#3FB950;font-weight:700"
                    if v == best else "" for v in col]
        elif col.name in ("RMSE", "MAE", "MAPE%"):  # lower is better
            best = col.min()
            return ["background-color:#1c3a2a;color:#3FB950;font-weight:700"
                    if v == best else "" for v in col]
        return [""] * len(col)

    styled = (metrics_df.style
              .apply(highlight_best, subset=["RMSE", "MAE", "R²", "MAPE%"])
              .format({"RMSE": "₹{:,.2f}", "MAE": "₹{:,.2f}",
                       "R²": "{:.4f}", "MAPE%": "{:.2f}%"}))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Bar chart comparison
    fig_cmp = go.Figure()
    for metric, color in [("RMSE", ACCENT), ("MAE", ACCENT2), ("R²", ACCENT3)]:
        fig_cmp.add_trace(go.Bar(
            name=metric, x=metrics_df["Model"], y=metrics_df[metric],
            marker_color=color, opacity=0.85,
        ))
    fig_cmp.update_layout(template=PLOTLY_TEMPLATE, barmode="group",
                           title="Model Comparison — RMSE · MAE · R²",
                           height=380)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:3rem;padding:16px 0;border-top:1px solid #21262D;
                text-align:center;color:#8B949E;font-size:0.78rem;">
        DSBDA Mini Project · E-Commerce Sales Forecasting ·
        Models: Linear Regression, Decision Tree, Random Forest, XGBoost
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
