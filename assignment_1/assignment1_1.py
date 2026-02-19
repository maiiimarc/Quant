# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:21:28 2026

@author: marco
"""

import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt


def download_prices_stooq(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
     """
     Download daily OHLCV from Stooq using pandas_datareader.
     For US stocks, Stooq often uses the format ’AAPL.US’.
     """
     start = pd.to_datetime(start_date)
     end = pd.to_datetime(end_date)
    
     candidates = [
     ticker,
     ticker.upper(),
     f"{ticker.upper()}.US",
     f"{ticker.upper()}.US"
     ]
    
     last_err = None
     for t in candidates:
         try:
             df = pdr.DataReader(t, "stooq", start, end)
             if df is not None and not df.empty:
                 # Stooq returns newest->oldest; sort to oldest->newest
                 df = df.sort_index()
                 df.index = pd.to_datetime(df.index)
                 return df
         except Exception as e:
             last_err = e

     raise RuntimeError(
         f"Failed to download data for {ticker} from Stooq. "
         f"Tried: {candidates}. Last error: {last_err}"
     )

 # -----------------------------
 # Parameters
 # -----------------------------
ticker = "AAPL"
start_date = "2010-01-01"
end_date = dt.date.today().strftime("%Y-%m-%d")

 # -----------------------------
 # Download data (Stooq)
 # -----------------------------
data = download_prices_stooq(ticker, start_date, end_date)

# Use Close (Stooq doesn’t always provide Adj Close)
#stock = data["Close"].astype(float)


# -----------------------------
# Compute Estimators (Formulas 8 & 13)
# -----------------------------

# 1. Setup Arrays for Vectorized Calculation
# Use 'Close' prices. Convert to numpy array for speed.
S = data["Close"].values 

# Define S_{t_k} (current prices) and S_{t_{k+1}} (next day prices)
S_tk   = S[:-1]
S_tkp1 = S[1:]

# N is the number of time intervals
N = len(S_tk)

# 2. Define Time Step (t_{k+1} - t_k)
# Standard convention: 1 trading day = 1/252 of a year. 
# This ensures mu and sigma are "annualized".
dt_k = 1.0 / 252.0

# 3. Calculate Drift Estimator: mu_hat (Formula 8)
# Formula: (1/N) * Sum( (1/dt) * (Return) )
simple_returns = (S_tkp1 - S_tk) / S_tk
mu_hat = (1 / N) * np.sum((1 / dt_k) * simple_returns)

# 4. Calculate Volatility Estimator: sigma^2_hat (Formula 13)
# Formula: (1/(N-1)) * Sum( (1/dt) * (Return - dt*mu)^2 )
term_inside = (simple_returns - (dt_k * mu_hat))**2
sigma2_hat = (1 / (N - 1)) * np.sum((1 / dt_k) * term_inside)

# Convert variance to standard deviation (volatility)
sigma_hat = np.sqrt(sigma2_hat)


# -----------------------------
# 5. Parkinson Volatility Estimator (Formula 2)
# -----------------------------
# Formula: (1 / 4*ln(2)) * [ln(High / Low)]^2
# We apply the same annualization logic (average daily variance * 252) 
# to make it comparable to the classical sigma above.

# Extract High and Low prices
H = data["High"].values
L = data["Low"].values

# Calculate the constant factor: 1 / (4 * ln(2))
const_parkinson = 1.0 / (4.0 * np.log(2.0))

# Compute daily logarithmic ranges squared
log_range_sq = (np.log(H / L))**2

# Compute Annualized Parkinson Variance
# Note: We use len(H) for the mean to include all data points available.
sigma2_parkinson = (1 / len(H)) * np.sum((1 / dt_k) * const_parkinson * log_range_sq)

# Compute Parkinson Volatility (Standard Deviation)
sigma_parkinson = np.sqrt(sigma2_parkinson)

# -----------------------------
# Output Results
# -----------------------------
print(f"Data range: {data.index[0].date()} to {data.index[-1].date()}")
print(f"Number of observations (N): {N}")
print("-" * 40)
print(f"{'Metric':<25} | {'Value':<10}")
print("-" * 40)
print(f"Drift (μ)                 | {mu_hat:.2%}")
print("-" * 40)
print(f"Classical Volatility (σ)  | {sigma_hat:.2%}")
print(f"Parkinson Volatility (σ)  | {sigma_parkinson:.2%}")
print("-" * 40)
print(f"Difference (Park - Class) | {sigma_parkinson - sigma_hat:.2%}")