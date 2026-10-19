
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="Crypto Analytics | BTC Forecast",
    page_icon="₿",
    layout="wide"
)

st.title("₿ Bitcoin Price Forecast")
st.write(
    "Historical Bitcoin analytics and ML-based price forecasting "
    "using hourly and daily market data."
)

st.caption(
    "Experimental forecasts only. Cryptocurrency prices are volatile, "
    "and predictions are not guaranteed."
)


# -------------------------------
# Download historical data
# -------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_btc_history(interval):
    data = yf.download(
        "BTC-USD",
        period="1y",
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No BTC data received for interval {interval}."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Close", "Volume"]].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    if len(data) < 100:
        raise ValueError(
            f"Insufficient historical data for {interval}."
        )

    return data


# -------------------------------
# Feature engineering
# -------------------------------

def make_features(data, interval):
    df = data.copy()

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    if interval == "1h":
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
        windows = [6, 24, 72, 168]
    else:
        lags = [1, 2, 3, 7, 14, 30, 60]
        windows = [7, 14, 30, 60]

    # Historical price returns
    for lag in lags:
        df[f"return_{lag}"] = close.pct_change(lag)

    # Rolling price statistics
    daily_return = close.pct_change()

    for window in windows:
        df[f"mean_{window}"] = (
            close.rolling(window).mean() / close - 1
        )

        df[f"volatility_{window}"] = (
            daily_return.rolling(window).std()
        )

        df[f"volume_mean_{window}"] = (
            volume / volume.rolling(window).mean() - 1
        )

    return df.replace(
        [np.inf, -np.inf], np.nan
    )


# -------------------------------
# Train and predict with ML
# -------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def predict_btc(interval, horizon, forecast_name):
    history = load_btc_history(interval)

    features = make_features(history, interval)

    # Target: future log return
    target = np.log(
        history["Close"].shift(-horizon) /
        history["Close"]
    ).rename("target")

    dataset = features.join(target).dropna()

    feature_columns = [
        col for col in dataset.columns
        if col not in ["Close", "target"]
    ]

    X = dataset[feature_columns]
    y = dataset["target"]

    if len(dataset) < 100:
        raise ValueError(
            f"Only {len(dataset)} training rows available. "
            "At least 100 are required."
        )

    # Keep the most recent observations relevant to this forecast.
    # The model is evaluated on a chronological holdout.
    split = int(len(dataset) * 0.80)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Holdout error in predicted log returns
    test_predictions = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_predictions)

    # Retrain on all available labelled data for the final forecast
    model.fit(X, y)

    latest_features = features[feature_columns].iloc[[-1]]

    if latest_features.isna().any().any():
        raise ValueError(
            "Latest features contain missing values."
        )

    predicted_log_return = float(
        model.predict(latest_features)[0]
    )

    current_price = float(history["Close"].iloc[-1])

    # Avoid exponential overflow from extreme model outputs
    predicted_log_return = float(
        np.clip(predicted_log_return, -2, 2)
    )

    predicted_price = current_price * np.exp(
        predicted_log_return
    )

    change_percent = (
        (predicted_price - current_price) / current_price
    ) * 100

    return {
        "name": forecast_name,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "change_percent": change_percent,
        "data_points": len(history),
        "training_rows": len(X),
        "test_mae": test_mae,
        "last_date": history.index[-1],
        "history": history
    }


# -------------------------------
# Historical data section
# -------------------------------

try:
    hourly_history = load_btc_history("1h")
    daily_history = load_btc_history("1d")

    latest_price = float(hourly_history["Close"].iloc[-1])
    previous_price = float(hourly_history["Close"].iloc[-2])

    hourly_change = (
        (latest_price - previous_price) / previous_price
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest BTC Price",
        f"${latest_price:,.2f}",
        f"{hourly_change:+.2f}%"
    )

    col2.metric(
        "Hourly Data Points",
        f"{len(hourly_history):,}"
    )

    col3.metric(
        "Daily Data Points",
        f"{len(daily_history):,}"
    )

    col4.metric(
        "Historical Period",
        "1 Year"
    )

    st.subheader("Bitcoin Historical Price")

    chart_choice = st.radio(
        "Select chart interval",
        ["Hourly", "Daily"],
        horizontal=True
    )

    if chart_choice == "Hourly":
        st.line_chart(hourly_history["Close"])
    else:
        st.line_chart(daily_history["Close"])

    st.subheader("Bitcoin Trading Volume")
    st.line_chart(daily_history["Volume"])

    st.caption(
        f"Latest hourly candle timestamp: {hourly_history.index[-1]}"
    )

except Exception as exc:
    st.error(f"Could not load historical data: {exc}")
    st.stop()


# -------------------------------
# Forecast section
# -------------------------------

st.divider()
st.subheader("🤖 ML Price Forecasts")

st.write(
    "Select a forecast horizon. The model uses historical returns, "
    "rolling price statistics, volatility and trading volume."
)

forecast_options = {
    "1 Hour": ("1h", 1),
    "1 Day": ("1h", 24),
    "1 Month": ("1d", 30),
    "3 Months": ("1d", 90)
}

selected_forecast = st.selectbox(
    "Forecast horizon",
    list(forecast_options.keys())
)

if st.button("Predict BTC Price", type="primary"):
    interval, horizon = forecast_options[selected_forecast]

    with st.spinner(
        f"Training Random Forest for {selected_forecast} forecast..."
    ):
        try:
            result = predict_btc(
                interval,
                horizon,
                selected_forecast
            )

            st.success(
                f"{selected_forecast} forecast calculated."
            )

            a, b, c = st.columns(3)

            a.metric(
                "Latest Price in Model Data",
                f"${result['current_price']:,.2f}"
            )

            b.metric(
                f"Predicted Price ({selected_forecast})",
                f"${result['predicted_price']:,.2f}",
                f"{result['change_percent']:+.2f}%"
            )

            c.metric(
                "Training Rows",
                f"{result['training_rows']:,}"
            )

            st.subheader("Forecast Summary")

            summary = pd.DataFrame({
                "Metric": [
                    "Forecast horizon",
                    "Data interval",
                    "Latest candle",
                    "Historical data points",
                    "Training rows",
                    "Holdout MAE (log return)"
                ],
                "Value": [
                    selected_forecast,
                    interval,
                    str(result["last_date"]),
                    result["data_points"],
                    result["training_rows"],
                    f"{result['test_mae']:.6f}"
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Historical Price and Forecast")

            history = result["history"]

            chart_data = history["Close"].tail(
                168 if interval == "1h" else 90
            ).copy()

            chart_data = chart_data.rename(
                "Historical BTC Price"
            )

            st.line_chart(chart_data)

            forecast_table = pd.DataFrame({
                "Metric": [
                    "Current model reference price",
                    "Estimated future price",
                    "Estimated percentage change"
                ],
                "Value": [
                    f"${result['current_price']:,.2f}",
                    f"${result['predicted_price']:,.2f}",
                    f"{result['change_percent']:+.2f}%"
                ]
            })

            st.dataframe(
                forecast_table,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "The forecast is a single estimated future price, "
                "not a live price or a guaranteed trading outcome. "
                "The holdout MAE measures error in log returns, "
                "not dollars. The model uses historical patterns "
                "and does not know future market news."
            )

        except Exception as exc:
            st.error(f"Forecast failed: {exc}")


# -------------------------------
# Footer
# -------------------------------

st.divider()

st.caption(
    "Crypto Analytics | Python • Streamlit • yfinance • "
    "Pandas • NumPy • Scikit-learn"
)
