# SCM_PredictiveModelling
Supply Chain Predictive Modeling Web App using Streamlit. Features LSTM demand forecasting, inventory optimization, customer segmentation, and statistical testing. Built with TensorFlow/Keras, Pandas, NumPy, Scikit‑learn, Plotly, and Statsmodels for interactive, data‑driven SCM insights.

SupplyChainManagement Modelling
├── .streamlit/
│   └── data/
│       ├── datagenerator.ipynb
│       └── ecommerce_supply_chain.csv"
├── streamlit_app/
│   ├── pages/
│       ├── 01_1️⃣forcastdemand.py
│       ├── 02_2️⃣inventoryoptimisation.py
│       ├── 03_3️⃣customer_segmentation.py
│       └── 04_4️⃣statisticaltests.py
│   
├── LICENSE
├── README.md
└── requirements.txt
1️⃣ Forecast Future Demand (LSTM Model)
What it does:
Leverages an LSTM (Long Short-Term Memory) neural network to predict future demand from historical sales data. Users can select an SKU to generate a detailed one-year demand forecast.

How it works:
The sales data is preprocessed to address non-stationarity and smooth fluctuations. The data is split into training (80%) and testing (20%) sets for robust validation. The LSTM model learns temporal patterns and forecasts future demand trends.

Output:

Interactive time-series plot showing historical sales, test predictions, and future forecasts.

Mean Absolute Percentage Error (MAPE) for model accuracy evaluation.

Downloadable forecast table for deeper analysis.

2️⃣ Inventory Optimisation & Demand Simulation
What it does:
Enables smart inventory planning using:

Economic Order Quantity (EOQ) – Calculates optimal order size to minimise total costs.

Safety Stock – Estimates buffer stock to avoid stockouts during demand fluctuations.

Monte Carlo Simulation – Models thousands of possible demand outcomes to measure uncertainty.

How it works:
Users pick an SKU to view its EOQ and safety stock. The simulation generates 1000+ demand variations using historical averages and variability to project risk.

Output:

EOQ & Safety Stock displayed in a summary table.

Demand distribution histogram for variability analysis.

Interactive visuals to assess trends and supply chain resilience.

3️⃣ Customer & Product Segmentation
What it does:
Delivers performance insights based on product movement and customer activity:

Sales by Supplier – Identify high-impact suppliers.

Sales by Product Family & Category – Spot best-performing items.

Top & Bottom Customers – Rank by revenue contribution.

Stock Turnover Ratio – Evaluate inventory movement efficiency.

How it works:
Aggregates sales, revenue, and stock levels across categories, suppliers, and products. Presents metrics through interactive charts and tables.

Output:

Top & bottom 20 SKUs by sales and turnover ratio.

Supplier-wise performance trends.

Customer segmentation by revenue.

4️⃣ Statistical Hypothesis Testing
What it does:
Performs five statistical tests to validate supply chain insights:

Pearson Correlation – Stock level vs sales.

Linear Regression – Lead time vs sales.

Kruskal-Wallis Test – Supplier revenue differences.

Spearman Correlation – Purchase frequency vs revenue.

ANOVA – Revenue variation across categories.

How it works:
Users choose a hypothesis from a dropdown, the app runs the test, and outputs p-values with interpretation.

Output:

Test statistic & p-value vs α = 0.05.

Decision to reject or retain H₀.

Optional visualisation of test distribution and rejection zones.

🎯 Why Use This App?
✔️ Uses a synthetic pharmaceutical supply chain dataset for realistic scenarios.
✔️ Integrates machine learning (LSTM) with data analytics for an end-to-end solution.
✔️ Covers demand forecasting, inventory planning, segmentation, and hypothesis testing.
✔️ Fully interactive via Streamlit with visualisations built using Plotly and statistical analysis with SciPy/Statsmodels.

🚀 Start by selecting a module from the sidebar.
