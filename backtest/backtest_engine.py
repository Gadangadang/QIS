# backtest/backtest_mean_reversion
import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yfinance as yf 
import pandas as pd 
import matplotlib.pyplot as plt 
from signals.mean_reversion import generate_mean_reversion_signal
from signals.momentum import generate_momentum_signal
from utils.metrics import *

def run_backtest(signal_func, df):
    # Filter rows from the beginning to end_date (inclusive)
    df = signal_func(df)
    
    df["Return"] = df["Close"].pct_change()
    df["Strategy"] = df["Position"].shift(1)*df["Return"]

    #Evaluation
    df[["Return", "Strategy"]].cumsum().plot(title="Cummulative returns")
    plt.show()

    sharpe = sharpe_ratio(df["Strategy"])
    drawdown = max_drawndown(df["Strategy"])
    print(f"Sharpe: {sharpe:.2f} | Drawdown: {drawdown:.2f}")
    
    
    
if __name__ == "__main__":
    df = pd.read_csv(
        "Dataset/spx_data_v1.csv",  # Replace with your CSV file path
        sep=';',          # Semicolon delimiter
        usecols=['Date', 'Open', 'Adj Close', 'Close', 'Volume'],  # Select desired columns
        index_col='Date', # Set Date as the index
        parse_dates=['Date'],  # Parse Date column as datetime
        dayfirst=True,    # Handle DD.MM.YYYY format (e.g., 02.01.2015)
        na_values=['#N/A N/A'],  # Treat '#N/A N/A' as NaN
    )
    # Convert Date index to datetime if not already parsed correctly
    df.index = pd.to_datetime(df.index, errors='coerce', dayfirst=True)
    # Drop rows with invalid dates (e.g., 1918 or numeric dates like 42009)
    df = df[df.index.notna()]
    
    
    run_backtest(generate_momentum_signal, df)
    
    
    