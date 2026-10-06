"""
Business Analytics Dashboard
-----------------------------
A pure-Python web app (Streamlit handles both the frontend UI and the
backend logic — no separate JS frontend needed) with two sidebar pages:

  Dashboard page — pick one or more "business points" to analyze:
    1. Monthly Sales Analysis
    2. Monthly Product Analysis
    3. Monthly Product Profit Analysis

  Profit Forecast page — trains several scikit-learn regression models
    (Linear, Ridge, Random Forest, Gradient Boosting) on historical
    Sales-type transactions to predict Profit, tunes hyperparameters
    with GridSearchCV, auto-selects the best-scoring model, shows its
    feature importances, and lets you predict profit for a new
    Product / Quantity / Year-Month / Territory / Vendor_Name combo.

joblib is used in two ways:
  - joblib.Memory: disk-caches the (potentially expensive) analysis
    and model-training functions so re-running the same query on the
    same data is instant.
  - joblib.dump/load: lets the user save a processed analysis result
    ("model"/summary object) to disk and reload it later without
    re-uploading or recomputing anything.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import hmac
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --------------------------------------------------------------------------
# Setup: joblib disk cache
# --------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / "cache"
SAVED_DIR = Path(__file__).parent / "saved_results"
CACHE_DIR.mkdir(exist_ok=True)
SAVED_DIR.mkdir(exist_ok=True)

memory = joblib.Memory(location=str(CACHE_DIR), verbose=0)

st.set_page_config(
    page_title="Business Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

BUSINESS_POINTS = [
    "Monthly Sales Analysis",
    "Monthly Product Analysis",
    "Monthly Product Profit Analysis",
]

REQUIRED_COLUMNS = ["Date", "Vendor_Name", "Territory", "Product", "Quantity", "Price_Pce", "Sales", "Type"]

# Profit-forecast ML feature schema
ML_CATEGORICAL_FEATURES = ["Purchased_Products", "Territory", "Vendor_Name"]
ML_NUMERIC_FEATURES = ["Quantity", "Year", "Month_Num"]
ML_FEATURES = ML_CATEGORICAL_FEATURES + ML_NUMERIC_FEATURES
ML_TARGET = "Profit"


# --------------------------------------------------------------------------
# Simple password gate
# --------------------------------------------------------------------------
def check_password() -> bool:
    """
    Shows a password prompt and returns True only once the correct
    password has been entered. The password is read from Streamlit
    secrets (st.secrets['app_password']) so it is never hard-coded into
    the source file — see the 'Setting the password' note in README.md.
    Uses hmac.compare_digest to avoid timing-attack leakage.
    """
    if st.session_state.get("authenticated", False):
        return True

    st.title("📊 Business Analytics Dashboard")
    st.subheader("🔒 Sign in")

    try:
        configured_password = st.secrets.get("app_password")
    except st.errors.StreamlitSecretNotFoundError:
        configured_password = None
    if not configured_password:
        st.error(
            "No password is configured for this app. Set `app_password` "
            "in .streamlit/secrets.toml (locally) or in the app's "
            "'Secrets' settings (Streamlit Community Cloud) before deploying."
        )
        st.stop()

    with st.form("login_form"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")

    if submitted:
        if hmac.compare_digest(entered, configured_password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    return False


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _read_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(buf)
    return pd.read_excel(buf)


@memory.cache
def load_and_clean(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load an uploaded file and normalize it into a standard schema.
    Cached by joblib: identical file bytes -> instant reload."""
    df = _read_any(file_bytes, filename)
    df.columns = [c.strip().title() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    # Note: both 'Sales' and 'Purchase' rows are kept here — Monthly Sales
    # and Monthly Product analysis filter down to 'Sales' rows themselves,
    # while Monthly Product Profit analysis needs both types.
    return df


def generate_sample_data(n_months: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Gizmo Z"]
    vendors = ["Acme Supplies", "Northwind Traders", "Globex Co"]
    territories = ["North", "South", "East", "West"]
    unit_price = {"Widget A": 25, "Widget B": 40, "Gadget X": 120,
                  "Gadget Y": 80, "Gizmo Z": 15}
    unit_cost = {p: round(v * rng.uniform(0.55, 0.75), 2) for p, v in unit_price.items()}

    rows = []
    start = pd.Timestamp.today().normalize().replace(day=1) - pd.DateOffset(months=n_months - 1)
    for m in range(n_months):
        month_start = start + pd.DateOffset(months=m)
        days_in_month = pd.Period(month_start, freq="M").days_in_month
        # Sales transactions
        for _ in range(60):
            day = rng.integers(1, days_in_month + 1)
            date = month_start.replace(day=int(day))
            product = rng.choice(products)
            qty = int(rng.integers(1, 20))
            price = unit_price[product]
            sales = round(qty * price * rng.uniform(0.9, 1.1), 2)
            rows.append([date, rng.choice(vendors), rng.choice(territories),
                         product, qty, price, sales, "Sales"])
        # Purchase (restocking) transactions
        for _ in range(20):
            day = rng.integers(1, days_in_month + 1)
            date = month_start.replace(day=int(day))
            product = rng.choice(products)
            qty = int(rng.integers(5, 40))
            cost = unit_cost[product]
            purchase_amt = round(qty * cost * rng.uniform(0.95, 1.05), 2)
            rows.append([date, rng.choice(vendors), rng.choice(territories),
                         product, qty, cost, purchase_amt, "Purchase"])

    df = pd.DataFrame(rows, columns=[
        "Date", "Vendor_Name", "Territory", "Product",
        "Quantity", "Price_Pce", "Sales", "Type",
    ])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df


# --------------------------------------------------------------------------
# Analysis functions (joblib-cached)
# --------------------------------------------------------------------------
@memory.cache
def monthly_sales_analysis(df: pd.DataFrame) -> dict:
    df = df[df["Type"] == "Sales"]
    monthly = (
        df.groupby("Month")
        .agg(Total_Sales=("Sales", "sum"), Orders=("Sales", "count"))
        .reset_index()
        .sort_values("Month")
    )
    monthly["MoM_Growth_%"] = monthly["Total_Sales"].pct_change().mul(100).round(2)
    monthly["Avg_Order_Value"] = (monthly["Total_Sales"] / monthly["Orders"]).round(2)

    summary = {
        "total_sales": float(df["Sales"].sum()),
        "total_orders": int(len(df)),
        "avg_monthly_sales": float(monthly["Total_Sales"].mean()),
        "best_month": monthly.loc[monthly["Total_Sales"].idxmax(), "Month"],
        "worst_month": monthly.loc[monthly["Total_Sales"].idxmin(), "Month"],
    }
    return {"table": monthly, "summary": summary}


@memory.cache
def monthly_product_analysis(df: pd.DataFrame) -> dict:
    df = df[df["Type"] == "Sales"]
    monthly_product = (
        df.groupby(["Month", "Product"])
        .agg(Total_Sales=("Sales", "sum"), Units_Sold=("Quantity", "sum"))
        .reset_index()
        .sort_values(["Month", "Total_Sales"], ascending=[True, False])
    )

    product_totals = (
        df.groupby("Product")
        .agg(Total_Sales=("Sales", "sum"), Units_Sold=("Quantity", "sum"))
        .reset_index()
        .sort_values("Total_Sales", ascending=False)
    )

    top_per_month = (
        monthly_product.sort_values(["Month", "Total_Sales"], ascending=[True, False])
        .groupby("Month")
        .first()
        .reset_index()[["Month", "Product", "Total_Sales"]]
        .rename(columns={"Product": "Top_Product", "Total_Sales": "Top_Product_Sales"})
    )

    summary = {
        "best_selling_product": product_totals.iloc[0]["Product"],
        "best_selling_product_total": float(product_totals.iloc[0]["Total_Sales"]),
        "num_products": int(df["Product"].nunique()),
    }
    return {
        "monthly_product": monthly_product,
        "product_totals": product_totals,
        "top_per_month": top_per_month,
        "summary": summary,
    }


@memory.cache
def monthly_product_profit_analysis(df: pd.DataFrame) -> dict:
    """
    Profit per product per month = (sum of Sales rows' Sales amount)
                                  - (sum of Purchase rows' Sales amount)
    """
    grouped = (
        df[df["Type"].isin(["Sales", "Purchase"])]
        .groupby(["Month", "Product", "Type"])["Sales"]
        .sum()
        .unstack("Type", fill_value=0)
        .reset_index()
    )
    for col in ("Sales", "Purchase"):
        if col not in grouped.columns:
            grouped[col] = 0.0

    grouped["Profit"] = grouped["Sales"] - grouped["Purchase"]
    grouped = grouped.rename(columns={"Sales": "Revenue", "Purchase": "Cost"})
    monthly_product_profit = grouped.sort_values(["Month", "Profit"], ascending=[True, False])

    product_profit_totals = (
        monthly_product_profit.groupby("Product")
        .agg(Total_Revenue=("Revenue", "sum"), Total_Cost=("Cost", "sum"), Total_Profit=("Profit", "sum"))
        .reset_index()
        .sort_values("Total_Profit", ascending=False)
    )

    monthly_profit_overall = (
        monthly_product_profit.groupby("Month")["Profit"]
        .sum()
        .reset_index()
        .sort_values("Month")
    )
    monthly_profit_overall["Cumulative_Profit"] = monthly_profit_overall["Profit"].cumsum()

    top_row = product_profit_totals.iloc[0] if not product_profit_totals.empty else None
    bottom_row = product_profit_totals.iloc[-1] if not product_profit_totals.empty else None
    summary = {
        "total_profit": float(monthly_product_profit["Profit"].sum()),
        "total_revenue": float(monthly_product_profit["Revenue"].sum()),
        "total_cost": float(monthly_product_profit["Cost"].sum()),
        "most_profitable_product": top_row["Product"] if top_row is not None else None,
        "most_profitable_product_total": float(top_row["Total_Profit"]) if top_row is not None else 0.0,
        "least_profitable_product": bottom_row["Product"] if bottom_row is not None else None,
        "least_profitable_product_total": float(bottom_row["Total_Profit"]) if bottom_row is not None else 0.0,
    }

    return {
        "monthly_product_profit": monthly_product_profit,
        "product_profit_totals": product_profit_totals,
        "monthly_profit_overall": monthly_profit_overall,
        "summary": summary,
    }


# --------------------------------------------------------------------------
# Profit Forecast — ML feature engineering
# --------------------------------------------------------------------------
@memory.cache
def build_profit_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a row-level training table for the profit-prediction models.

    Target (Profit) is computed per Sales transaction as:
        Profit = Sales revenue - (Quantity * allocated unit purchase cost)
    where the unit purchase cost for a Product is estimated from
    Type == 'Purchase' rows for that Product in the same Year-Month
    (falling back to that Product's overall average purchase cost, and
    finally to the dataset-wide average purchase cost, if a given
    month has no purchase record for that product).

    Features:
        Purchased_Products  - the product (categorical)
        Quantity             - quantity for that sales transaction
        Year, Month_Num       - parsed from the Year-Month ('Month')
        Territory             - categorical
        Vendor_Name           - categorical
    """
    sales = df[df["Type"] == "Sales"].copy()
    purchases = df[df["Type"] == "Purchase"].copy()

    # Unit cost per (Month, Product), from purchase records
    monthly_cost = (
        purchases.groupby(["Month", "Product"])
        .agg(_cost=("Sales", "sum"), _qty=("Quantity", "sum"))
        .reset_index()
    )
    monthly_cost["Unit_Cost"] = monthly_cost["_cost"] / monthly_cost["_qty"].replace(0, np.nan)

    # Fallback 1: product-level average unit cost across all months
    product_cost = (
        purchases.groupby("Product")
        .agg(_cost=("Sales", "sum"), _qty=("Quantity", "sum"))
        .reset_index()
    )
    product_cost["Unit_Cost_Fallback"] = product_cost["_cost"] / product_cost["_qty"].replace(0, np.nan)

    # Fallback 2: dataset-wide average unit cost
    overall_unit_cost = (
        purchases["Sales"].sum() / purchases["Quantity"].replace(0, np.nan).sum()
        if not purchases.empty else 0.0
    )
    if pd.isna(overall_unit_cost):
        overall_unit_cost = 0.0

    sales = sales.merge(monthly_cost[["Month", "Product", "Unit_Cost"]], on=["Month", "Product"], how="left")
    sales = sales.merge(product_cost[["Product", "Unit_Cost_Fallback"]], on="Product", how="left")
    sales["Unit_Cost"] = sales["Unit_Cost"].fillna(sales["Unit_Cost_Fallback"]).fillna(overall_unit_cost)
    sales = sales.drop(columns=["Unit_Cost_Fallback"])

    sales["Cost_Allocated"] = sales["Quantity"] * sales["Unit_Cost"]
    sales["Profit"] = sales["Sales"] - sales["Cost_Allocated"]

    sales["Year"] = sales["Month"].str.slice(0, 4).astype(int)
    sales["Month_Num"] = sales["Month"].str.slice(5, 7).astype(int)
    sales["Purchased_Products"] = sales["Product"]

    feature_df = sales[
        ML_CATEGORICAL_FEATURES + ML_NUMERIC_FEATURES + [ML_TARGET, "Month", "Product"]
    ].dropna(subset=ML_FEATURES + [ML_TARGET])

    return feature_df.reset_index(drop=True)


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ML_CATEGORICAL_FEATURES),
            ("num", "passthrough", ML_NUMERIC_FEATURES),
        ]
    )


@memory.cache
def train_profit_models(feature_df: pd.DataFrame) -> dict:
    """
    Trains and hyperparameter-tunes several regressors to predict Profit,
    picks the best by held-out test R^2, and extracts feature importances
    for the winning model (aggregated back to the original feature names).
    """
    X = feature_df[ML_FEATURES]
    y = feature_df[ML_TARGET]

    n_splits = 3 if len(feature_df) >= 30 else 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_specs = {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge Regression": (Ridge(random_state=42), {"model__alpha": [0.1, 1.0, 10.0]}),
        "Random Forest": (
            RandomForestRegressor(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 6, 12],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }

    results = []
    fitted_pipelines = {}
    for name, (estimator, grid) in model_specs.items():
        pipe = Pipeline([("prep", _build_preprocessor()), ("model", estimator)])
        if grid:
            search = GridSearchCV(pipe, grid, cv=n_splits, scoring="r2", n_jobs=-1)
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            best_params = search.best_params_
            cv_r2 = float(search.best_score_)
        else:
            pipe.fit(X_train, y_train)
            best_pipe = pipe
            best_params = {}
            cv_r2 = float(np.mean(
                [r2_score(y_train, best_pipe.predict(X_train))]
            ))

        preds = best_pipe.predict(X_test)
        test_r2 = float(r2_score(y_test, preds))
        test_mae = float(mean_absolute_error(y_test, preds))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

        results.append({
            "Model": name,
            "Best_Params": str(best_params) if best_params else "(default)",
            "CV_R2": round(cv_r2, 4),
            "Test_R2": round(test_r2, 4),
            "Test_MAE": round(test_mae, 2),
            "Test_RMSE": round(test_rmse, 2),
        })
        fitted_pipelines[name] = best_pipe

    results_df = pd.DataFrame(results).sort_values("Test_R2", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_model_name]

    # Feature importances for the winning model, aggregated to original feature names
    prep = best_pipeline.named_steps["prep"]
    model = best_pipeline.named_steps["model"]
    encoded_cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(ML_CATEGORICAL_FEATURES))
    all_feature_names = encoded_cat_names + ML_NUMERIC_FEATURES

    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_importances = np.abs(model.coef_)
    else:
        raw_importances = np.zeros(len(all_feature_names))

    imp_map = {feat: 0.0 for feat in ML_FEATURES}
    for fname, val in zip(all_feature_names, raw_importances):
        matched = next((orig for orig in ML_CATEGORICAL_FEATURES if fname.startswith(orig + "_")), None)
        key = matched if matched else fname
        imp_map[key] = imp_map.get(key, 0.0) + float(val)

    importance_df = (
        pd.DataFrame({"Feature": list(imp_map.keys()), "Importance": list(imp_map.values())})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    total_importance = importance_df["Importance"].sum()
    if total_importance > 0:
        importance_df["Importance_%"] = (importance_df["Importance"] / total_importance * 100).round(2)
    else:
        importance_df["Importance_%"] = 0.0

    test_predictions = pd.DataFrame({
        "Actual_Profit": y_test.values,
        "Predicted_Profit": best_pipeline.predict(X_test),
    })

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "importance_df": importance_df,
        "test_predictions": test_predictions,
    }


def predict_profit(pipeline, purchased_product, quantity, year, month_num, territory, vendor_name) -> float:
    row = pd.DataFrame([{
        "Purchased_Products": purchased_product,
        "Quantity": quantity,
        "Year": year,
        "Month_Num": month_num,
        "Territory": territory,
        "Vendor_Name": vendor_name,
    }])[ML_FEATURES]
    return float(pipeline.predict(row)[0])


# --------------------------------------------------------------------------
# UI sections
# --------------------------------------------------------------------------
def show_monthly_sales(df: pd.DataFrame):
    st.header("📈 Monthly Sales Analysis")
    result = monthly_sales_analysis(df)
    table, summary = result["table"], result["summary"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales", f"{summary['total_sales']:,.2f}")
    c2.metric("Total Orders", f"{summary['total_orders']:,}")
    c3.metric("Avg Monthly Sales", f"{summary['avg_monthly_sales']:,.2f}")

    fig = px.bar(table, x="Month", y="Total_Sales", title="Total Sales by Month",
                 text_auto=".2s")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(table, x="Month", y="MoM_Growth_%", markers=True,
                    title="Month-over-Month Growth (%)")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Details")
    st.dataframe(table, use_container_width=True)

    st.caption(
        f"Best month: **{summary['best_month']}** · "
        f"Worst month: **{summary['worst_month']}**"
    )

    return result


def show_monthly_product(df: pd.DataFrame):
    st.header("📦 Monthly Product Analysis")
    result = monthly_product_analysis(df)
    monthly_product = result["monthly_product"]
    product_totals = result["product_totals"]
    top_per_month = result["top_per_month"]
    summary = result["summary"]

    c1, c2 = st.columns(2)
    c1.metric("Best-Selling Product", summary["best_selling_product"],
               f"{summary['best_selling_product_total']:,.2f} total sales")
    c2.metric("Distinct Products", summary["num_products"])

    fig = px.bar(product_totals, x="Product", y="Total_Sales",
                 title="Total Sales by Product", text_auto=".2s")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(monthly_product, x="Month", y="Total_Sales", color="Product",
                  title="Monthly Sales by Product (stacked)", barmode="stack")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top Product per Month")
    st.dataframe(top_per_month, use_container_width=True)

    st.subheader("Full Detail")
    st.dataframe(monthly_product, use_container_width=True)

    return result


def show_monthly_product_profit(df: pd.DataFrame):
    st.header("💰 Monthly Product Profit Over Time")
    st.caption("Profit = Sales revenue (Type = 'Sales') − Purchase cost (Type = 'Purchase'), by product.")
    result = monthly_product_profit_analysis(df)
    monthly_product_profit = result["monthly_product_profit"]
    product_profit_totals = result["product_profit_totals"]
    monthly_profit_overall = result["monthly_profit_overall"]
    summary = result["summary"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Profit", f"{summary['total_profit']:,.2f}")
    c2.metric("Total Revenue", f"{summary['total_revenue']:,.2f}")
    c3.metric("Total Cost", f"{summary['total_cost']:,.2f}")

    if summary["most_profitable_product"] is not None:
        c4, c5 = st.columns(2)
        c4.metric("Most Profitable Product", summary["most_profitable_product"],
                   f"{summary['most_profitable_product_total']:,.2f} profit")
        c5.metric("Least Profitable Product", summary["least_profitable_product"],
                   f"{summary['least_profitable_product_total']:,.2f} profit")

    fig = px.line(monthly_product_profit, x="Month", y="Profit", color="Product",
                  markers=True, title="Profit by Product Over Time")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(monthly_profit_overall, x="Month", y="Cumulative_Profit",
                   markers=True, title="Cumulative Profit Over Time (All Products)")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(product_profit_totals, x="Product", y="Total_Profit",
                  title="Total Profit by Product", text_auto=".2s")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Profit by Product per Month")
    st.dataframe(monthly_product_profit, use_container_width=True)

    st.subheader("Profit Totals by Product")
    st.dataframe(product_profit_totals, use_container_width=True)

    return result


def show_profit_forecast_page(df: pd.DataFrame):
    st.header("🔮 Profit Forecast")
    st.write(
        "Trains several regression models on historical Sales transactions "
        "to predict **Profit**, tunes each model's hyperparameters, and "
        "automatically picks the best-scoring one based on held-out test "
        "R² — then shows what that model learned (feature importance) and "
        "lets you predict profit for a hypothetical order."
    )

    feature_df = build_profit_ml_dataset(df)

    if len(feature_df) < 15:
        st.warning(
            f"Only {len(feature_df)} usable Sales rows with matching purchase-cost "
            "data were found — that's too few to train a reliable model. "
            "Upload a larger dataset with both 'Sales' and 'Purchase' type rows."
        )
        return

    with st.spinner("Training and tuning models..."):
        ml_result = train_profit_models(feature_df)

    results_df = ml_result["results_df"]
    best_model_name = ml_result["best_model_name"]
    best_pipeline = ml_result["best_pipeline"]
    importance_df = ml_result["importance_df"]
    test_predictions = ml_result["test_predictions"]

    st.subheader("Model Comparison")
    st.caption(
        "Each model tuned via cross-validated grid search where applicable; "
        "ranked by Test R² (higher is better; 1.0 = perfect fit)."
    )

    def _highlight_best(row):
        color = "background-color: #C6EFCE" if row["Model"] == best_model_name else ""
        return [color] * len(row)

    st.dataframe(results_df.style.apply(_highlight_best, axis=1), use_container_width=True)
    st.success(f"🏆 Best model: **{best_model_name}** (Test R² = {results_df.iloc[0]['Test_R2']:.3f})")

    c1, c2 = st.columns(2)
    with c1:
        fig_imp = px.bar(
            importance_df, x="Importance_%", y="Feature", orientation="h",
            title=f"Feature Importance — {best_model_name}",
        )
        fig_imp.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_imp, use_container_width=True)
    with c2:
        fig_scatter = px.scatter(
            test_predictions, x="Actual_Profit", y="Predicted_Profit",
            title="Actual vs Predicted Profit (test set)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.subheader("Predict Profit for a New Order")

    products = sorted(feature_df["Purchased_Products"].unique().tolist())
    territories = sorted(feature_df["Territory"].unique().tolist())
    vendors = sorted(feature_df["Vendor_Name"].unique().tolist())
    years = sorted(feature_df["Year"].unique().tolist())
    next_year = max(years) if years else datetime.now().year

    with st.form("predict_profit_form"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            sel_product = st.selectbox("Purchased Product", products)
            sel_quantity = st.number_input("Quantity", min_value=1, value=10, step=1)
        with pc2:
            sel_year = st.number_input("Year", min_value=2000, max_value=2100, value=int(next_year), step=1)
            sel_month = st.selectbox("Month", list(range(1, 13)), index=0,
                                       format_func=lambda m: f"{m:02d}")
        with pc3:
            sel_territory = st.selectbox("Territory", territories)
            sel_vendor = st.selectbox("Vendor Name", vendors)

        predict_clicked = st.form_submit_button("Predict Profit")

    if predict_clicked:
        pred = predict_profit(
            best_pipeline, sel_product, sel_quantity, sel_year, sel_month,
            sel_territory, sel_vendor,
        )
        st.metric("Predicted Profit", f"{pred:,.2f}")
        st.caption(
            f"Prediction from **{best_model_name}** for {sel_product}, qty {sel_quantity}, "
            f"{sel_year}-{sel_month:02d}, {sel_territory}, vendor {sel_vendor}."
        )


# --------------------------------------------------------------------------
# Save / load processed results with joblib
# --------------------------------------------------------------------------
def save_results(name: str, payload: dict):
    path = SAVED_DIR / f"{name}.joblib"
    joblib.dump(payload, path)
    return path


def list_saved_results():
    return sorted(SAVED_DIR.glob("*.joblib"))


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
def main():
    if not check_password():
        return

    if st.sidebar.button("🚪 Log out"):
        st.session_state["authenticated"] = False
        st.rerun()

    st.sidebar.header("📁 Navigate")
    page = st.sidebar.radio(
        "Page", ["Dashboard", "Profit Forecast"], label_visibility="collapsed"
    )

    st.sidebar.header("Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload sales data (CSV or Excel)", type=["csv", "xlsx"]
    )
    st.sidebar.caption(
        "Expected columns: **Date, Vendor_Name, Territory, Product, "
        "Quantity, Price_Pce, Sales, Type** (Type = 'Sales' or 'Purchase')"
    )
    use_sample = st.sidebar.button("Use sample data instead")

    df = None
    if uploaded_file is not None:
        try:
            df = load_and_clean(uploaded_file.getvalue(), uploaded_file.name)
        except ValueError as e:
            st.error(str(e))
            return
    elif use_sample or st.session_state.get("use_sample_data"):
        st.session_state["use_sample_data"] = True
        df = generate_sample_data()
        st.info("Using generated sample data.")

    if df is None:
        st.title("📊 Business Analytics Dashboard")
        st.info("Upload a file or click **Use sample data instead** in the sidebar to begin.")
        return

    if page == "Profit Forecast":
        show_profit_forecast_page(df)
        return

    # ---- Dashboard page ----
    st.title("📊 Business Analytics Dashboard")
    st.write(
        "Select one or more business points, provide your data, and get "
        "instant analysis. Results are cached and can be saved with `joblib` "
        "for later reuse."
    )

    st.sidebar.header("Select Business Points")
    selected = st.sidebar.multiselect(
        "Business points to analyze",
        BUSINESS_POINTS,
        default=[BUSINESS_POINTS[0]],
    )

    if not selected:
        st.warning("Select at least one business point from the sidebar.")
        return

    with st.expander("Preview raw data"):
        st.dataframe(df.head(50), use_container_width=True)

    results_to_save = {}

    if "Monthly Sales Analysis" in selected:
        results_to_save["monthly_sales_analysis"] = show_monthly_sales(df)
        st.divider()

    if "Monthly Product Analysis" in selected:
        results_to_save["monthly_product_analysis"] = show_monthly_product(df)
        st.divider()

    if "Monthly Product Profit Analysis" in selected:
        results_to_save["monthly_product_profit_analysis"] = show_monthly_product_profit(df)
        st.divider()

    # Save / load section
    st.sidebar.header("Save / Load Results (joblib)")
    save_name = st.sidebar.text_input("Save current results as", value="latest_run")
    if st.sidebar.button("💾 Save with joblib"):
        path = save_results(save_name, results_to_save)
        st.sidebar.success(f"Saved to {path.name}")

    saved = list_saved_results()
    if saved:
        pick = st.sidebar.selectbox(
            "Load a previous run", [p.name for p in saved]
        )
        if st.sidebar.button("📂 Load selected"):
            loaded = joblib.load(SAVED_DIR / pick)
            st.sidebar.success(f"Loaded {pick} — see below")
            st.header(f"Loaded result: {pick}")
            for key, val in loaded.items():
                st.subheader(key.replace("_", " ").title())
                if "table" in val:
                    st.dataframe(val["table"])
                if "monthly_product" in val:
                    st.dataframe(val["monthly_product"])
                if "monthly_product_profit" in val:
                    st.dataframe(val["monthly_product_profit"])
                st.json(val["summary"])


if __name__ == "__main__":
    main()
