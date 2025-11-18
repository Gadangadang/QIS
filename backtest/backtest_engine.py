# backtest/backtest_mean_reversion
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from signals.momentum import MomentumSignal
from signals.mean_reversion import MeanReversionSignal
from signals.ensemble import EnsembleSignal
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
    
    trader.print_summary()
    
    trader.plot()
    
    return {
        "train_df": train_signals,
        "test_input_df": test_signals_all,
        "test_result_df": result_df,
        "trader": trader,
        "summary": summary,
    }



if __name__ == "__main__":
    print("Loading and FIXING SPX daily data...")

    df = pd.read_csv("Dataset/spx_full_1990_2025.csv", index_col=0)  # load as strings first

    # THIS IS THE NUCLEAR FIX — forces datetime index no matter what
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")

    # Drop any impossible dates (should be none)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])  # also removes bad rows

    # Sort chronological (oldest first)
    df = df.sort_index(ascending=True)

    # Final check — THIS WILL NOW WORK
    print(f"Loaded {len(df):,} daily bars")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Index type: {type(df.index)} → {type(df.index[0])}")
    
    # Final sanity check
    print(f"Columns: {list(df.columns)}")
    print(f"First 3 rows:\n{df.head(3)}")
    print(f"Last 3 rows:\n{df.tail(3)}")

    
    
    signals_to_test = {
        "Momentum_60d": MomentumSignal(lookback=60,  threshold=0.025),
        "Momentum_120d": MomentumSignal(lookback=120, threshold=0.02),
        "Momentum_250d": MomentumSignal(lookback=250, threshold=0.018),
        "MR_10": MeanReversionSignal(window=10,  entry_z=2.2, exit_z=1.0),
        "MR_20": MeanReversionSignal(window=20,  entry_z=2.0, exit_z=1.0),
        "MR_60": MeanReversionSignal(window=60,  entry_z=1.8, exit_z=1.0),
        "Ensemble": EnsembleSignal(),  # ← this uses the fixed signals inside
    }

    for name, signal in signals_to_test.items():
        print(f"\n=== Testing {name} ===")
        result = run_train_test(signal.generate, df.copy())
        print(f"→ Test Sharpe: {result['summary']['sharpe']:.2f} | Return: {result['summary']['total_return_pct']:.1%}")
