# backtest/backtest_mean_reversion
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignal
from utils.metrics import max_drawdown, sharpe_ratio
from live.paper_trader import PaperTrader


def split_train_test(df: pd.DataFrame, train_frac: float = 0.6, lookback: int = 20):
    """Split dataframe into train and test windows, returning the train slice and a
    test slice that includes `lookback` rows of history before the test start.

    Returns (train_df, test_with_history_df, test_only_df)
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")

    n = len(df)
    train_n = int(n * train_frac)
    train_df = df.iloc[:train_n].copy()
    hist_start = max(0, train_n - lookback)
    test_with_history = df.iloc[hist_start:].copy()
    test_only = df.iloc[train_n:].copy()
    return train_df, test_with_history, test_only


def run_train_test(signal_func, df: pd.DataFrame, train_frac: float = 0.6, lookback: int = 20, initial_cash: float = 100000):
    """Run a train/test split. The signal function is applied to the train slice for analysis,
    then re-applied over the test window including `lookback` history so indicators can initialize.

    Returns a dict with train_df, test_input_df (with history), test_result_df and trader summary.
    """
    train_df, test_with_history, test_only = split_train_test(df, train_frac=train_frac, lookback=lookback)

    # Generate signals on train and evaluate
    train_signals = signal_func(train_df)
    train_signals["Return"] = train_signals["Close"].pct_change()
    train_signals["Strategy"] = train_signals["Position"].shift(1) * train_signals["Return"]

    # Simple train evaluation
    train_sharpe = sharpe_ratio(train_signals["Strategy"].fillna(0))
    train_mdd = max_drawdown(train_signals["Strategy"].fillna(0))

    print(f"Train period: Sharpe {train_sharpe:.2f} | MaxDD {train_mdd:.2%}")

    # Generate signals on the test window but include history for lookback
    test_signals_all = signal_func(test_with_history)

    # Start date is the first index of test_only
    start_date = test_only.index[0]

    # Run paper trading on the full test_with_history but only count metrics/trades from start_date
    trader = PaperTrader(initial_cash=initial_cash)
    result_df = trader.simulate(test_signals_all, position_col="Position", start_date=start_date)

    summary = trader.summary()
    print(f"Test period summary: {summary}")

    return {
        "train_df": train_signals,
        "test_input_df": test_signals_all,
        "test_result_df": result_df,
        "trader": trader,
        "summary": summary,
    }


def run_backtest(signal_func, df):
    # Filter rows from the beginning to end_date (inclusive)
    df = signal_func(df)

    df["Return"] = df["Close"].pct_change()
    df["Strategy"] = df["Position"].shift(1) * df["Return"]

    # Evaluation
    df[["Return", "Strategy"]].cumsum().plot(title="Cummulative returns")
    plt.show()

    sharpe = sharpe_ratio(df["Strategy"])
    # max_drawdown expects daily returns (not cumulative)
    drawdown = max_drawdown(df["Strategy"]) 
    print(f"Sharpe: {sharpe:.2f} | Max Drawdown: {drawdown:.2%}")
    return df


if __name__ == "__main__":
    df = pd.read_csv(
        "Dataset/spx_data_v1.csv",  # Replace with your CSV file path
        sep=";",  # Semicolon delimiter
        usecols=[
            "Date",
            "Open",
            "Adj Close",
            "Close",
            "Volume",
        ],  # Select desired columns
        index_col="Date",  # Set Date as the index
        parse_dates=["Date"],  # Parse Date column as datetime
        dayfirst=True,  # Handle DD.MM.YYYY format (e.g., 02.01.2015)
        na_values=["#N/A N/A"],  # Treat '#N/A N/A' as NaN
    )
    # Convert Date index to datetime if not already parsed correctly
    df.index = pd.to_datetime(df.index, errors="coerce", dayfirst=True)
    # Drop rows with invalid dates (e.g., 1918 or numeric dates like 42009)
    df = df[df.index.notna()]

    mom = MeanReversionSignal()
    run_train_test(mom.generate, df)
