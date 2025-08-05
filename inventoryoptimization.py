import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import scipy.stats as stats

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Inventory Optimization", page_icon="📦", layout="wide")

# --------- USER SETTINGS (Update if your csv columns differ) ------------
SKU_COL = "SKU"
SALES_COL = "Sales Quantity"
LEAD_TIME_COL = "Lead Time (days)"
CATEGORY_COL = "Category"  # optional

# --------- Load dataset ------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    # Drop rows with missing critical data
    df_clean = df.dropna(subset=[SKU_COL, SALES_COL, LEAD_TIME_COL])
    return df_clean

data_path = r"C:\Users\elroy\OneDrive\Desktop\Supplychain\ecommerce_supply_chain.csv"
df = load_data(data_path)

st.title("📦 Inventory Optimization Tool")

# -------- Sidebar Inputs for Cost Parameters ----------
st.sidebar.header("Settings")

ordering_cost = st.sidebar.number_input(
    "Ordering Cost per order (€)", min_value=0.0, value=50.0, step=1.0,
    help="Cost incurred every time an order is placed"
)
holding_cost = st.sidebar.number_input(
    "Holding Cost per unit per year (€)", min_value=0.0, value=2.0, step=0.1,
    help="Cost to hold one unit in inventory for a year"
)
service_level = st.sidebar.slider(
    "Service Level (Confidence for Safety Stock)", min_value=0.80, max_value=0.99,
    value=0.95, step=0.01,
    help="Probability of not running out of stock"
)

# ---------- Aggregations per SKU ----------
df_grouped = df.groupby(SKU_COL).agg(
    avg_demand=(SALES_COL, "mean"),
    std_demand=(SALES_COL, "std"),
    avg_lead_time=(LEAD_TIME_COL, "mean")
).reset_index()

# Fill any NaN std_dev with 0 (e.g., constant demand)
df_grouped["std_demand"] = df_grouped["std_demand"].fillna(0)

# ---------- EOQ Calculation ----------
def calculate_eoq(demand, ordering_cost, holding_cost):
    if demand <= 0 or holding_cost == 0:
        return 0
    return round(np.sqrt((2 * demand * ordering_cost) / holding_cost), 2)

df_grouped["EOQ"] = df_grouped.apply(
    lambda row: calculate_eoq(row["avg_demand"], ordering_cost, holding_cost),
    axis=1
)

# ---------- Safety Stock Calculation ----------
def calculate_safety_stock(std_dev_demand, lead_time_days, service_level=0.95):
    z_score = stats.norm.ppf(service_level)
    if std_dev_demand <= 0 or lead_time_days <= 0:
        return 0
    return round(z_score * std_dev_demand * np.sqrt(lead_time_days), 2)

df_grouped["Safety Stock"] = df_grouped.apply(
    lambda row: calculate_safety_stock(row["std_demand"], row["avg_lead_time"], service_level),
    axis=1
)

# ---------- Optional: Category-level Inventory ----------
if CATEGORY_COL in df.columns:
    st.markdown("### Category-level Inventory Summary")
    df_cat = df.groupby(CATEGORY_COL).agg(
        avg_demand=(SALES_COL, "mean"),
        std_demand=(SALES_COL, "std"),
        avg_lead_time=(LEAD_TIME_COL, "mean")
    ).reset_index()
    df_cat["std_demand"] = df_cat["std_demand"].fillna(0)
    df_cat["EOQ"] = df_cat.apply(
        lambda row: calculate_eoq(row["avg_demand"], ordering_cost, holding_cost),
        axis=1
    )
    df_cat["Safety Stock"] = df_cat.apply(
        lambda row: calculate_safety_stock(row["std_demand"], row["avg_lead_time"], service_level),
        axis=1
    )
    st.dataframe(df_cat.set_index(CATEGORY_COL)[["EOQ", "Safety Stock"]])

# ---------- Show SKU Inventory Table ----------
st.markdown("### Inventory Parameters per SKU")
st.dataframe(df_grouped.set_index(SKU_COL)[["EOQ", "Safety Stock"]])

# ---------- Monte Carlo Simulation Function ----------
def monte_carlo_simulation(avg_demand, std_dev, days=30, sims=1000):
    # Generate random demand based on normal distribution (clip at 0)
    simulated = np.random.normal(loc=avg_demand, scale=std_dev, size=(sims, days))
    simulated = np.clip(simulated, a_min=0, a_max=None)
    means = simulated.mean(axis=1)
    return simulated, means

# ---------- Monte Carlo Simulation UI ----------
st.markdown("### Monte Carlo Simulation of Future Demand")

sku_list = df_grouped[SKU_COL].tolist()
selected_sku = st.selectbox("Select SKU to simulate demand:", sku_list)

if selected_sku:
    sku_data = df_grouped[df_grouped[SKU_COL] == selected_sku].iloc[0]
    st.write(f"Average demand: {sku_data['avg_demand']:.2f}")
    st.write(f"Demand Std Dev: {sku_data['std_demand']:.2f}")
    st.write(f"Average Lead Time (days): {sku_data['avg_lead_time']:.2f}")

    days_sim = st.number_input("Days to simulate:", min_value=7, max_value=90, value=30, step=1)
    sims = st.number_input("Number of simulations:", min_value=100, max_value=10000, value=1000, step=100)

    if st.button("Run Simulation"):
        simulated_demands, means = monte_carlo_simulation(
            sku_data["avg_demand"], sku_data["std_demand"], days=days_sim, sims=sims
        )
        fig = px.histogram(
            means,
            nbins=30,
            title=f"Monte Carlo Simulation: Avg Demand Distribution for {selected_sku}",
            labels={"value": "Average Demand"},
            opacity=0.75,
        )
        fig.update_layout(xaxis_title="Simulated Average Demand", yaxis_title="Frequency", template="plotly_white")
        st.plotly_chart(fig)

        st.success(f"Simulation completed for SKU **{selected_sku}**!")

