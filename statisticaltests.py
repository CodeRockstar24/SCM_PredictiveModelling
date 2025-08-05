import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Statistical Tests", page_icon="🧮", layout="wide")

# -------------------- LOAD DATA --------------------
df = pd.read_csv(
    r"C:\Users\elroy\OneDrive\Desktop\Supplychain\ecommerce_supply_chain.csv",
    parse_dates=["Date"]
)

# -------------------- PARAMETERS --------------------
alpha = 0.05

# -------------------- SIDEBAR --------------------
st.sidebar.header("Select Statistical Test")
test_options = [
    "Stockouts reduce sales (Pearson Correlation)",
    "Longer lead times lower sales (Linear Regression)",
    "Supplier revenue distribution (Kruskal-Wallis Test)",
    "Frequent buyers generate higher revenue (Spearman Correlation)",
    "Certain categories generate higher revenue (ANOVA)",
    "Category vs Returns (Chi-Square Test)",
    "Warehouse revenue difference (Mann-Whitney U Test)",
    "Sales distribution normality (Kolmogorov–Smirnov Test)",
    "Lead time difference between two suppliers (T-Test)"
]
selected_test = st.sidebar.selectbox("Choose a test to perform:", test_options)

# -------------------- TEST LOGIC --------------------

if selected_test == "Stockouts reduce sales (Pearson Correlation)":
    st.title("Stockouts Reduce Sales (Pearson Correlation)")
    pearson_corr, p_val = stats.pearsonr(df["Stock Level"], df["Sales Quantity"])
    decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"
    st.write(f"Correlation Coefficient (r): {pearson_corr:.4f}")
    st.write(f"P-value: {p_val:.4f}")
    st.write(f"Decision: {decision}")

elif selected_test == "Longer lead times lower sales (Linear Regression)":
    st.title("Longer Lead Times Lower Sales (Linear Regression)")
    X = df["Lead Time (days)"].values.reshape(-1, 1)
    y = df["Sales Quantity"].values
    model = LinearRegression().fit(X, y)
    r_sq = model.score(X, y)
    p_val = stats.pearsonr(df["Lead Time (days)"], df["Sales Quantity"])[1]
    decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"
    st.write(f"R²: {r_sq:.4f}")
    st.write(f"P-value: {p_val:.4f}")
    st.write(f"Decision: {decision}")

elif selected_test == "Supplier revenue distribution (Kruskal-Wallis Test)":
    st.title("Supplier Revenue Distribution (Kruskal-Wallis Test)")
    groups = [df[df["Supplier"] == s]["Revenue (USD)"] for s in df["Supplier"].unique()]
    stat, p_val = stats.kruskal(*groups)
    decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"
    st.write(f"Kruskal-Wallis Statistic: {stat:.4f}")
    st.write(f"P-value: {p_val:.4f}")
    st.write(f"Decision: {decision}")

elif selected_test == "Frequent buyers generate higher revenue (Spearman Correlation)":
    st.title("Frequent Buyers Generate Higher Revenue (Spearman Correlation)")
    rfm = df.groupby("Customer ID").agg({"Customer ID": "count", "Revenue (USD)": "sum"})
    rfm.columns = ["Frequency", "Monetary"]
    corr, p_val = stats.spearmanr(rfm["Frequency"], rfm["Monetary"])
    decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"
    st.write(f"Spearman Correlation (ρ): {corr:.4f}")
    st.write(f"P-value: {p_val:.4f}")
    st.write(f"Decision: {decision}")

elif selected_test == "Certain categories generate higher revenue (ANOVA)":
    st.title("Certain Categories Generate Higher Revenue (ANOVA)")
    groups = [df[df["Category"] == c]["Revenue (USD)"] for c in df["Category"].unique()]
    stat, p_val = stats.f_oneway(*groups)
    decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"
    st.write(f"F-Statistic: {stat:.4f}")
    st.write(f"P-value: {p_val:.4f}")
    st.write(f"Decision: {decision}")

elif selected_test == "Category vs Returns (Chi-Square Test)":
    st.title("Category vs Returns (Chi-Square Test)")
    contingency_table = pd.crosstab(df["Category"], df["Return Quantity"] > 0)
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"
    st.write(f"Chi-Square Statistic: {chi2:.4f}")
    st.write(f"P-value: {p_val:.4f}")
    st.write(f"Decision: {decision}")

elif selected_test == "Warehouse revenue difference (Mann-Whitney U Test)":
    st.title("Warehouse Revenue Difference (Mann-Whitney U Test)")
    warehouses = df["Warehouse Location"].unique()
    if len(warehouses) >= 2:
        group1 = df[df["Warehouse Location"] == warehouses[0]]["Revenue (USD)"]
        group2 = df[df["Warehouse Location"] == warehouses[1]]["Revenue (USD)"]
        stat, p_val = stats.mannwhitneyu(group1, group2)
        decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"
        st.write(f"Mann-Whitney U Statistic: {stat:.4f}")
        st.write(f"P-value: {p_val:.4f}")
        st.write(f"Decision: {decision}")
    else:
        st.write("Not enough warehouse locations for comparison.")

elif selected_test == "Sales distribution normality (Kolmogorov–Smirnov Test)":
    st.title("Sales Distribution Normality (Kolmogorov–Smirnov Test)")
    sales = df["Sales Quantity"]
    stat, p_val = stats.kstest(sales, 'norm', args=(sales.mean(), sales.std()))
    decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"
    st.write(f"KS Statistic: {stat:.4f}")
    st.write(f"P-value: {p_val:.4f}")
    st.write(f"Decision: {decision}")

elif selected_test == "Lead time difference between two suppliers (T-Test)":
    st.title("Lead Time Difference Between Two Suppliers (T-Test)")
    suppliers = df["Supplier"].unique()
    if len(suppliers) >= 2:
        group1 = df[df["Supplier"] == suppliers[0]]["Lead Time (days)"]
        group2 = df[df["Supplier"] == suppliers[1]]["Lead Time (days)"]
        stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"
        st.write(f"T-Statistic: {stat:.4f}")
        st.write(f"P-value: {p_val:.4f}")
        st.write(f"Decision: {decision}")
    else:
        st.write("Not enough suppliers for comparison.")
