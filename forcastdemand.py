import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Forecast Future Demand", page_icon="📊", layout="wide")

# Load dataset
df = pd.read_csv(
    r"C:\Users\elroy\OneDrive\Desktop\Supplychain\ecommerce_supply_chain.csv",
    parse_dates=["Date"]
)

# -------------------- SIDEBAR --------------------
st.sidebar.header("Forecast Settings")

# Forecast granularity
forecast_level = st.sidebar.selectbox("Forecast by:", ["SKU", "Category", "Supplier"])

# Unique list based on chosen granularity
if forecast_level == "SKU":
    options = df["SKU"].unique()
elif forecast_level == "Category":
    options = df["Category"].unique()
else:
    options = df["Supplier"].unique()

selected_option = st.sidebar.selectbox(f"Choose {forecast_level}:", options)

# Adjustable time_steps for LSTM sequence length
time_steps = st.sidebar.slider("Time Steps (LSTM Sequence Length)", min_value=4, max_value=52, value=24, step=1)

target_column = "Sales Quantity"  # fixed for demand forecasting

# Instructions
st.sidebar.markdown(
    """
    ### Instructions:
    1. Select forecast level (SKU, Category, or Supplier).
    2. Pick an item from the dropdown.
    3. Adjust time steps if needed.
    4. Click **Train Model and Forecast**.
    """
)

# -------------------- FORECAST BUTTON --------------------
if st.sidebar.button("Train Model and Forecast"):
    # Filter data based on selection
    df_filtered = df[df[forecast_level] == selected_option]

    # Aggregate sales over time
    df_forecast = df_filtered.groupby("Date")[target_column].sum().reset_index()
    df_forecast.set_index("Date", inplace=True)

    # -------------------- STATIONARITY CHECK --------------------
    def check_stationarity(timeseries):
        result = adfuller(timeseries)
        return timeseries if result[1] <= 0.05 else timeseries.diff().dropna()

    df_forecast[target_column] = check_stationarity(df_forecast[target_column])
    df_forecast = df_forecast.resample('W').sum()  # Weekly aggregation
    df_forecast["Rolling Sales"] = df_forecast[target_column].rolling(window=4, min_periods=1).mean()
    df_forecast.dropna(subset=["Rolling Sales"], inplace=True)

    # -------------------- SCALING --------------------
    scaler = MinMaxScaler()
    df_forecast["Scaled"] = scaler.fit_transform(df_forecast[["Rolling Sales"]])

    # Train-Test Split
    split_idx = int(len(df_forecast) * 0.8)
    train_data = df_forecast.iloc[:split_idx]
    test_data = df_forecast.iloc[split_idx:]

    # Check for minimum data length
    min_required_points = time_steps + 5  # buffer

    if len(train_data) < min_required_points or len(test_data) < min_required_points:
        st.error(
            f"Not enough data points to forecast {forecast_level}: {selected_option}. "
            f"Need at least {min_required_points} points in train and test sets, "
            f"but got {len(train_data)} (train) and {len(test_data)} (test). "
            "Try selecting another option or reducing time steps."
        )
        st.stop()

    # Create LSTM sequences
    def create_lstm_data(series, ts):
        X, y = [], []
        for i in range(len(series) - ts):
            X.append(series[i: i + ts])
            y.append(series[i + ts])
        return np.array(X), np.array(y)

    train_series = train_data["Scaled"].values
    test_series = test_data["Scaled"].values

    X_train, y_train = create_lstm_data(train_series, time_steps)
    X_test, y_test = create_lstm_data(test_series, time_steps)

    if X_train.size == 0 or X_test.size == 0:
        st.error(f"Time steps ({time_steps}) too large for dataset size. Try lowering time steps.")
        st.stop()

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # -------------------- LSTM MODEL --------------------
    @st.cache_resource
    def build_lstm():
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(time_steps, 1)),
            Dropout(0.3),
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    lstm_model = build_lstm()
    lstm_model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)

    # -------------------- PREDICTIONS --------------------
    lstm_predictions = lstm_model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    pred_series_lstm = pd.Series(lstm_predictions.flatten(), index=test_data.index[time_steps:])

    mape_lstm = mean_absolute_percentage_error(test_data["Rolling Sales"].iloc[time_steps:], pred_series_lstm)

    # -------------------- FUTURE FORECAST --------------------
    forecast_steps = 52
    lstm_forecast = []
    last_data = test_data["Scaled"].values[-time_steps:]

    for _ in range(forecast_steps):
        input_data = last_data.reshape((1, time_steps, 1))
        pred = lstm_model.predict(input_data)[0, 0]
        lstm_forecast.append(pred)
        last_data = np.append(last_data[1:], pred)

    lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1))
    forecast_dates = pd.date_range(test_data.index[-1], periods=forecast_steps + 1, freq='W')[1:]
    pred_series_future = pd.Series(lstm_forecast.flatten(), index=forecast_dates)

    # -------------------- PLOTTING --------------------
    st.subheader(f"LSTM Forecast for {forecast_level}: {selected_option}")
    st.metric("Forecast Accuracy (MAPE)", f"{mape_lstm * 100:.2f} %")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_data.index, y=train_data["Rolling Sales"],
        mode='lines', name='Train Data', line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=test_data.index, y=test_data["Rolling Sales"],
        mode='lines', name='Test Data', line=dict(color='green', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pred_series_lstm.index, y=pred_series_lstm,
        mode='lines', name='Predictions', line=dict(color='red', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=pred_series_future.index, y=pred_series_future,
        mode='lines', name='Future Forecast', line=dict(color='purple', dash='dot')
    ))

    fig.update_layout(
        title=f"1-Year LSTM Forecast for {forecast_level}: {selected_option}",
        xaxis_title="Date", yaxis_title=target_column,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )

    st.plotly_chart(fig)

    # -------------------- SHOW FORECAST TABLE --------------------
    st.subheader("Forecasted Sales Data")
    forecast_df = pd.DataFrame({"Date": pred_series_future.index, "Predicted Sales": pred_series_future.values})
    st.dataframe(forecast_df.set_index("Date"))
