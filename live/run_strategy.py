#live / run_strategy.py
import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yfinance as yf 
import pandas as pd    
from datetime import date  
from signals.mean_reversion import generate_mean_reversion_signal

from retrying import retry

# Define a retry decorator for rate limit errors
@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=2)
def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        raise ValueError(f"No data retrieved for {ticker}")
    return data


def main():
    # Configuration
    ticker = "SPX"
    start_date = "2015-01-01"
    end_date = "2025-11-12"
    cache_file = f"Dataset/{ticker}_data.csv"
    # Step 1: Download data with retry
    try:
        df = download_data(ticker, start=start_date, end=end_date)
        df.to_csv(cache_file)
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
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
        
    print(df)
   
    # Step 2: Process data (simplified example)
    if not df.empty:
        # Example: Add dummy columns (replace with your actual strategy logic)
        df = generate_mean_reversion_signal(df)  # Replace with your position calculation
        df["Return"] = df["Close"].pct_change()  # Example return calculation
        df["Strategy"] = df["Return"] * df["Signal"]  # Example strategy return

        # Step 3: Extract today's data safely
        try:
            today_data = df.iloc[-1][["Signal", "Position", "Return", "Strategy"]]
            print("Today's data:", today_data)
        except (IndexError, KeyError) as e:
            print(f"Error accessing data: {e}")
            today_data = None
    else:
        print("No data available. Skipping strategy execution.")
        today_data = None

    # Step 4: Proceed with today_data (e.g., execute trades)
    if today_data is not None:
        today_data = df.iloc[-1][["Signal", "Position", "Return", "Strategy"]]
        today_data["Date"] = date.today()
        log_path = f"logs/daily_log.csv"
        
        try:
            log_df = pd.read_csv("log_path")
        except:
            log_df = pd.DataFrame()
            
        log_df = pd.concat([log_df, pd.DataFrame([today_data])])
        log_df.to_csv(log_path, index=False)

if __name__ == "__main__":
    main()
    
