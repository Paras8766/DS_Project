` # E-Commerce Sales Forecasting

A beginner-friendly project for preprocessing, analysis, forecasting, and dashboarding of e-commerce sales data.

## Project Structure

```text
ecommerce-forecast/
├── data/
│   ├── amazon_sales.csv
│   └── cleaned_sales.csv
├── outputs/
│   └── plots/
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── models.py
│   └── dashboard.py
├── requirements.txt
└── README.md
```

## Dataset Columns Used

- Date
- Status
- Category
- Qty
- Amount
- ship-state
- B2B

## Setup

1. Create virtual environment

```powershell
python -m venv venv
```

2. Activate virtual environment

Windows:

```powershell
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Run Steps

### Step 1: Data Preprocessing

```powershell
python src/data_preprocessing.py
```

Output:

- `data/cleaned_sales.csv`

### Step 2: EDA Plots

```powershell
python src/eda.py
```

Outputs in `outputs/plots/`:

- `monthly_revenue.png`
- `top_5_categories_revenue.png`
- `top_10_states_revenue.png`
- `b2b_vs_b2c_revenue.png`

### Step 3: Model Training and Forecasting

```powershell
python src/models.py
```

Outputs in `outputs/plots/`:

- `random_forest_feature_importance.png`

Console output includes:

- Linear Regression metrics (RMSE, MAE, R²)
- Random Forest metrics (RMSE, MAE, R²)
- Final model comparison table

### Step 4: Streamlit Dashboard

```powershell
streamlit run src/dashboard.py
```

Dashboard includes:

- Sidebar filters: Category, State, Date range
- KPIs: Total Revenue, Total Orders, Avg Order Value
- Monthly revenue trend
- Category and state revenue charts
- Model metrics comparison table

## Notes

- Keep `data/amazon_sales.csv` in the `data/` folder before running scripts.
- Scripts are intentionally simple and beginner-friendly.
