import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd 
from datetime import date 
from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignal
from backtest.backtest_engine import run_train_test
from utils.logger import log_daily_result
from live.paper_trader import PaperTrader
import os
import pandas as pd


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
    
    MOM_signal = MomentumSignal(50, entry_z=2, exit_z=0.5)
    MR_signal = MeanReversionSignal()

    # Generate signals
    df = signal.generate(df)

    # Run paper trading simulation
    trader = PaperTrader(initial_cash=100_000)
    result_df = trader.simulate(df, position_col="Position")

    # Persist trade log if available
    trades = getattr(trader, "trades", None)
    if trades is not None and not trades.empty:
        os.makedirs("logs", exist_ok=True)
        trades_path = os.path.join("logs", "trades_spx.csv")
        trades.to_csv(trades_path, index=False)
        print(f"Saved trades to {trades_path}")

    # Print summary and show plots
    trader.print_summary()
    
    trader.plot()

    # Log daily result (keeps compatibility with existing logger)
    # Ensure Strategy column exists for logging
    if "Strategy" not in result_df.columns:
        result_df["Return"] = result_df["Close"].pct_change()
        result_df["Strategy"] = result_df["ExecPosition"] * result_df["Return"]

    log_daily_result(result_df, "SPX Index")
    
