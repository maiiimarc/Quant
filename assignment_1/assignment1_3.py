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
from skfolio.datasets import load_sp500_implied_vol_dataset



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
window_size = 30  # 30-day rolling window

 # -----------------------------
 # Download data (Stooq)
 # -----------------------------
data = download_prices_stooq(ticker, start_date, end_date)

# Use Close (Stooq doesn’t always provide Adj Close)
#stock = data["Close"].astype(float)


# -----------------------------
# 2. Compute Realized Estimators
# -----------------------------

# Classical Volatility (Standard Deviation of Returns)
data['Return'] = data['Close'].pct_change()
data['Realized_Classical'] = data['Return'].rolling(window=window_size).std() * np.sqrt(252)

# Parkinson Volatility (High-Low Range)
# Formula: sqrt( Mean( Daily_Parkinson_Variance ) * 252 )
# Step A: Calculate the daily contribution to Parkinson variance: (1/4ln2) * ln(H/L)^2
# Step B: Average this daily variance over the rolling window
# Note: We use .mean() because the formula is a Sum divided by N (average)
# Step C: Annualize and sqrt
const_parkinson = 1.0 / (4.0 * np.log(2.0))
data['Log_Range_Sq'] = (np.log(data['High'] / data['Low']))**2
#data['Realized_Parkinson'] = np.sqrt(data['Log_Range_Sq'].rolling(window=window_size).sum() * const_parkinson)
data['Realized_Parkinson'] = np.sqrt(data['Log_Range_Sq'].rolling(window=window_size).mean() * const_parkinson) * np.sqrt(252)

# 3. Garman-Klass Estimator (OHLC)
log_hl = np.log(data['High'] / data['Low'])**2
log_co = np.log(data['Close'] / data['Open'])**2
data['GK_Var'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

data['Realized_GK'] = np.sqrt(data['GK_Var'].rolling(window=window_size).mean()) * np.sqrt(252)

# -----------------------------
# 3. Retrieve Implied Volatility (skfolio)
# -----------------------------

iv_dataset = load_sp500_implied_vol_dataset()

data = data.join(iv_dataset[ticker].rename("Implied_Vol"))

# -----------------------------
# 4. Output & Plotting
# -----------------------------
# Filter data to where we have valid values for a cleaner plot
plot_data = data.dropna(subset=['Realized_Classical', 'Realized_Parkinson', 'Realized_GK'])

print(f"Data range: {plot_data.index[0].date()} to {plot_data.index[-1].date()}")
print("-" * 40)
print(plot_data[['Realized_Classical', 'Realized_Parkinson', 'Implied_Vol']].tail())

plt.figure(figsize=(12, 6))

# Plot Realized Volatilities
plt.plot(plot_data.index, plot_data['Realized_Classical'], 
         label=f'Realized Classical ({window_size}d)', linewidth=1.2, alpha=0.6)
plt.plot(plot_data.index, plot_data['Realized_Parkinson'], 
         label=f'Realized Parkinson ({window_size}d)', linewidth=1.2, alpha=0.6)
plt.plot(plot_data.index, plot_data['Realized_GK'],
         label=f'Realized_GK ({window_size}d)', linewidth=1.2, alpha=0.6)

# Plot Implied Volatility 
plt.plot(plot_data.index, plot_data['Implied_Vol'], 
             label='Implied Volatility (Market Expectation)', 
             linewidth=1.2, linestyle='--', color='black', alpha=0.7)

plt.title(f'Realized vs. Implied Volatility: {ticker}')
plt.ylabel('Annualized Volatility')
plt.xlabel('Date')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(f'volatility_comparison_{ticker}.png', dpi=300)
plt.show()