import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd 
from datetime import date 
from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignal
from backtest.backtest_engine import run_backtest
from utils.logger import log_daily_result


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
    
    signal = MomentumSignal()
    
    df = run_backtest(signal.generate, df)
    
    log_daily_result(df, "SPX Index")