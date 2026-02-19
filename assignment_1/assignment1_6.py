# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 13:43:33 2026

@author: marco

Combined Analysis: Realized Volatility Estimators (SPX) vs. VIX
Merged and expanded from assignment1_3.py and vix1_5.py
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# -----------------------------
# 1. Data Fetching Functions
# -----------------------------

def download_spx_ohlc_stooq(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily OHLCV for S&P 500 from Stooq.
    Stooq ticker for S&P 500 is usually '^SPX' or 'SPX.US'.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # List of likely tickers for SPX on Stooq
    candidates = ["^SPX", "SPX.US"]
    
    last_err = None
    for t in candidates:
        try:
            df = pdr.DataReader(t, "stooq", start, end)
            if df is not None and not df.empty:
                df = df.sort_index()
                df.index = pd.to_datetime(df.index)
                # Ensure all columns are numeric
                for col in ['Open', 'High', 'Low', 'Close']:
                     df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
        except Exception as e:
            last_err = e

    print(f"Warning: Could not fetch SPX OHLC. Last error: {last_err}")
    return pd.DataFrame()

def fetch_vix_close(start_date, end_date):
    """
    Fetch daily VIX close series.
    Primary: FRED VIXCLS (Cleanest source)
    Fallback: Stooq
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Try FRED first
    try:
        df = pdr.DataReader("VIXCLS", "fred", start, end)
        df = df.rename(columns={"VIXCLS": "VIX_Close"}).dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        print("FRED VIX fetch failed, trying Stooq...")
    
    # Fallback: Stooq
    for sym in ["VI.F", "^VIX"]:
        try:
            df = pdr.DataReader(sym, "stooq", start, end)
            df = df.sort_index()
            df.index = pd.to_datetime(df.index)
            if "Close" in df.columns and not df.empty:
                return df[["Close"]].rename(columns={"Close": "VIX_Close"}).dropna()
        except Exception:
            continue
    return pd.DataFrame()

# -----------------------------
# 2. Main Execution & Calculations
# -----------------------------

# Parameters
ticker = "SPX"
start_date = "2015-01-01"
end_date = dt.date.today().strftime("%Y-%m-%d")
window_size = 30  # 30-day rolling window for realized vol

print(f"Fetching data from {start_date} to {end_date}...")

# A. Get Data
spx_data = download_spx_ohlc_stooq(start_date, end_date)
vix_data = fetch_vix_close(start_date, end_date)

if spx_data.empty or vix_data.empty:
    raise RuntimeError("Failed to download necessary data (SPX or VIX). Check internet or tickers.")

# B. Compute Realized Estimators on SPX
# Note: VIX is annualized, so we must annualize realized vol (multiply by sqrt(252))

# 1. Classical Volatility (Std Dev of Close-to-Close returns)
spx_data['Return'] = spx_data['Close'].pct_change()
spx_data['Realized_Classical'] = spx_data['Return'].rolling(window=window_size).std() * np.sqrt(252) * 100

# 2. Parkinson Volatility (High-Low Range)
# Formula: sqrt( 1/(4ln2) * mean(ln(H/L)^2) )
const_parkinson = 1.0 / (4.0 * np.log(2.0))
spx_data['Log_Range_Sq'] = (np.log(spx_data['High'] / spx_data['Low']))**2
spx_data['Realized_Parkinson'] = np.sqrt(spx_data['Log_Range_Sq'].rolling(window=window_size).mean() * const_parkinson) * np.sqrt(252) * 100

# 3. Garman-Klass Estimator (OHLC)
# Uses Open, High, Low, Close for better efficiency than Parkinson
log_hl = np.log(spx_data['High'] / spx_data['Low'])**2
log_co = np.log(spx_data['Close'] / spx_data['Open'])**2
spx_data['GK_Var'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
spx_data['Realized_GK'] = np.sqrt(spx_data['GK_Var'].rolling(window=window_size).mean()) * np.sqrt(252) * 100

# C. Merge Data
# We align the indices. VIX is already in percentage points (e.g., 20.0), 
# so we multiplied realized vols by 100 above to match scale.
combined = spx_data.join(vix_data, how='inner')

# Drop NaNs created by rolling windows
analysis_df = combined.dropna(subset=['Realized_Classical', 'Realized_Parkinson', 'Realized_GK', 'VIX_Close'])

# -----------------------------
# 3. Statistical Analysis
# -----------------------------
print("\n" + "="*40)
print("Analysis of Correlation & Cointegration")
print("="*40)

# A. Correlation
corr_matrix = analysis_df[['Realized_Classical', 'Realized_Parkinson', 'Realized_GK', 'VIX_Close']].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# B. Cointegration Test (Engle-Granger)
# Null Hypothesis: Series are NOT cointegrated.
# Low p-value (< 0.05) implies cointegration (long-run relationship exists).
print("\n[Cointegration] Realized GK vs VIX:")
score, p_value, _ = coint(analysis_df['Realized_GK'], analysis_df['VIX_Close'])
print(f"t-statistic: {score:.4f}")
print(f"p-value: {p_value:.4f} ({'Cointegrated' if p_value < 0.05 else 'Not Cointegrated'})")

# -----------------------------
# 4. Plotting
# -----------------------------
plt.figure(figsize=(14, 7))

# Plot Realized Estimators
plt.plot(analysis_df.index, analysis_df['Realized_Classical'], 
         label=f'Realized Classical ({window_size}d)', linewidth=1, alpha=0.6)
plt.plot(analysis_df.index, analysis_df['Realized_GK'],
         label=f'Realized Garman-Klass ({window_size}d)', linewidth=1.2, alpha=0.8, color='green')

# Plot VIX (Implied)
plt.plot(analysis_df.index, analysis_df['VIX_Close'], 
         label='VIX (Market Implied)', 
         linewidth=1.5, linestyle='-', color='black', alpha=0.8)

plt.title('S&P 500: Realized Volatility Estimators vs. VIX')
plt.ylabel('Annualized Volatility (%)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(f'vix_comparison_{ticker}.png', dpi=300)
plt.show()