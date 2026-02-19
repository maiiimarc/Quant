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
window_size = 30  # 30-day rolling window

 # -----------------------------
 # Download data (Stooq)
 # -----------------------------
data = download_prices_stooq(ticker, start_date, end_date)

# Use Close (Stooq doesn’t always provide Adj Close)
stock = data["Close"].astype(float)

