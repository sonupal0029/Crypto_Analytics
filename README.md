# Crypto Analytics — Bitcoin Price Forecast

A Streamlit application for exploring historical Bitcoin (BTC-USD) market data and generating experimental machine-learning price forecasts with a Random Forest regressor.

> **Disclaimer:** Cryptocurrency markets are volatile. Forecasts are experimental estimates, not financial advice or guaranteed outcomes.

## Features

- Downloads Bitcoin historical data from Yahoo Finance using `yfinance`.
- Shows latest available hourly price, hourly change, and hourly/daily data counts.
- Visualizes historical Bitcoin closing prices and trading volume.
- Builds features from historical returns, rolling price statistics, volatility, and volume.
- Trains a `RandomForestRegressor` and evaluates it on a chronological holdout split.
- Provides selectable forecast horizons:
  - 1 hour
  - 1 day
  - 1 month
  - 3 months
- Displays the estimated price, percentage change, training-row count, and holdout MAE for log returns.

## Project structure

```text
Crypto_Analytics/
├── app.py
├── cryptodata.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE                 # Optional; add only if you choose a license
```

## Requirements

- Python 3.10 or newer recommended
- Git (for version control and GitHub upload)
- Internet connection for Yahoo Finance data retrieval

## Setup locally

### 1. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run commands using the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run the application

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## How it works

1. The app retrieves BTC-USD price and volume data.
2. It cleans numeric data and handles Yahoo Finance multi-level column names.
3. It creates lagged percentage-return features and rolling statistics.
4. It creates a target based on the log return between the current close and the selected future horizon.
5. It trains a Random Forest model on the first 80% of the labelled observations and measures mean absolute error on the remaining chronological holdout.
6. It retrains on all labelled rows and estimates the future price from the latest feature row.

## Important implementation notes

- Yahoo Finance availability and rate limits can affect data retrieval.
- Forecasts are based on historical patterns and do not account for future news or events.
- The app reports MAE in **log-return units**, not dollars.
- The hourly data endpoint may provide a shorter history than daily data depending on Yahoo Finance's current rules.
- The notebook `cryptodata.ipynb` is included as a companion analysis file. Review its outputs and remove private or unnecessary content before publishing.
- The provided app trains its model during a forecast request. The files `random_forest_model.pkl` and `scaler.pkl` are not required by the visible `app.py` code. Keep them out of Git unless another project file uses them and you intentionally want to publish them.

## Troubleshooting

### `ModuleNotFoundError`

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

### No BTC data received / Yahoo Finance errors

Check your internet connection, retry later, and verify that Yahoo Finance is responding. Data access is an external dependency.

### GitHub push rejected

Check the repository URL and authentication. If the GitHub repository was initialized with its own README, the remote may contain a commit not present locally; reconcile the histories before pushing rather than force-pushing blindly.

## Suggested future improvements

- Add forecast confidence ranges and compare them with a simple baseline.
- Save model evaluation metrics and plot predicted vs. actual returns.
- Add a date range selector and downloadable historical data.
- Add automated tests for feature generation and forecasting.
- Pin tested dependency versions once the app runs successfully in a clean environment.

## Disclaimer

This project is for learning and demonstration purposes only. It is not investment advice. Do not make trading decisions based only on model outputs.
