# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 23:16:41 2026

@author: marco
"""
import numpy as np
import pandas as pd
import datetime
import math
from pandas_datareader import data as pdr


def fetch_spx_close(start_date, end_date):
    """
    Fetch an SPX-like daily close series.
    Primary: FRED SP500
    Fallback: Stooq ˆSPX
    Returns a DataFrame with a ’Close’ column and DatetimeIndex.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    # Try FRED first
    try:
        df = pdr.DataReader("SP500", "fred", start, end)
        df = df.rename(columns={"SP500": "Close"}).dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        pass
    
    # Fallback: Stooq
    df = pdr.DataReader("ˆSPX", "stooq", start, end)
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    return df[["Close"]].dropna()


def fetch_vix_close(start_date, end_date):
    """
    Fetch daily VIX close series.
    Primary: FRED VIXCLS
    Fallback: Stooq VI.F (VIX)
    Returns a DataFrame with a ’Close’ column and DatetimeIndex.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    # Try FRED first
    try:
        df = pdr.DataReader("VIXCLS", "fred", start, end)
        df = df.rename(columns={"VIXCLS": "Close"}).dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        pass
    
    # Fallback: Stooq (try common symbols)
    for sym in ["VI.F", "vi.f", "ˆVIX", "ˆvix"]:
        try:
            df = pdr.DataReader(sym, "stooq", start, end)
            df = df.sort_index()
            df.index = pd.to_datetime(df.index)
            if "Close" in df.columns and not df.empty:
                return df[["Close"]].dropna()
        except Exception:
                continue
    raise ValueError("No VIX data found from FRED or Stooq.")


spx_symbol = "ˆSPX"
today = pd.to_datetime("2025-03-05") # Keep this fixed in your implementation
end_date = today
start_date = end_date- datetime.timedelta(days=365)

spx_data = fetch_spx_close(start_date, end_date)
if spx_data.empty:
   raise ValueError("No SPX data found from FRED/Stooq for this window.")

lastBusDay = spx_data.index[-1]
S0 = float(spx_data["Close"].iloc[-1]) # spot/closing price

# Fetch VIX in a short window after lastBusDay (first available close)
vix_data = fetch_vix_close(lastBusDay, lastBusDay + datetime.timedelta(days=30))
if vix_data.empty:
    raise ValueError("No VIX data found from FRED/Stooq for this window.")
vix_market = float(vix_data["Close"].iloc[0])

# Fixed inputs (keep these fixed in your implementation)
#T = 27/365.0
T = 30/365.0
r = 0.02
F0 = S0 * math.exp(r * T) # forward approximation

print("Last Bus Day:", lastBusDay)
print("S0:", S0)
print("VIX market close:", vix_market)
print("F0:", F0)



# import pandas as pd
# from yahooquery import Ticker

# spx_ticker = Ticker("ˆSPX")

# # Option chain table (may be empty depending on data availability)
# oc = spx_ticker.option_chain

# if oc is None or (isinstance(oc, pd.DataFrame) and oc.empty):
#     print("No option chain returned. If needed, use the CSVs provided on Canvas.")
# else:
#     # Available expirations are contained in the MultiIndex level "expiration"
#     expirations = oc.index.get_level_values("expiration").unique()
#     expirations = sorted(pd.to_datetime(expirations).strftime("%Y-%m-%d").tolist())
#     print("Available expirations:", expirations)



print("COMPUTING ESTIMATED VIX")

# 1. Load Data & Calculate Mid Prices
calls = pd.read_csv("Call_option_data_2025-04-03_final.csv")
puts = pd.read_csv("Put_option_data_2025-04-03_final.csv")

calls['mid'] = (calls['bid'] + calls['ask']) / 2.0
puts['mid']  = (puts['bid'] + puts['ask']) / 2.0


# 3. Split into Puts and Calls based on F0
# For puts, we use strikes < F0. For calls, strikes > F0.
puts_sub = puts[puts['strike'] < F0].sort_values('strike')
calls_sub = calls[calls['strike'] > F0].sort_values('strike')
#  compare directly to F0 here


# Extract arrays for fast iteration
put_strikes = puts_sub['strike'].values
put_prices  = puts_sub['mid'].values

call_strikes = calls_sub['strike'].values
call_prices  = calls_sub['mid'].values

# 4. Compute Put Summation (Left Riemann Sum)
# Sum_{i=1}^{n_p-1} P(K_i) * (1/K_i - 1/K_{i+1})
put_sum = 0.0
for i in range(len(put_strikes) - 1):
    K_i = put_strikes[i]
    K_next = put_strikes[i+1]
    P_i = put_prices[i]
    put_sum += P_i * (1.0 / K_i - 1.0 / K_next)

# 5. Compute Call Summation (Right Riemann Sum)
# Sum_{i=1}^{n_c} C(K_i) * (1/K_{i-1} - 1/K_i)
call_sum = 0.0
for i in range(1, len(call_strikes)):
    K_prev = call_strikes[i-1]
    K_i = call_strikes[i]
    C_i = call_prices[i]
    call_sum += C_i * (1.0 / K_prev - 1.0 / K_i)

# 6. Final VIX Calculation (incorporating the scalar 2*e^(r*tau)/tau)
# Note: tau is T in our script
vix_squared = (2.0 * math.exp(r * T) / T) * (put_sum + call_sum)

vix_estimated = 100 * math.sqrt(vix_squared)

print(f"Computed VIXt: {vix_estimated:.4f}")
print(f"Market VIX   : {vix_market:.4f}")