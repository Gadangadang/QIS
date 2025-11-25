"""Sanity check: synthetic trending data to verify signal+trader pipeline.

Run:
    python3 -m scripts.sanity_check_signal

This creates a gentle uptrend with noise; the `MomentumSignal` should generate
some long positions and the PaperTrader should produce positive small returns
when transaction costs are zero.
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from signals.momentum import MomentumSignal
from core.paper_trader import PaperTrader


def make_trend(n=500, start_price=1000.0, drift=0.0005, vol=0.005):
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    closes = [start_price]
    for i in range(1, n):
        ret = drift + np.random.randn() * vol
        closes.append(closes[-1] * (1 + ret))
    df = pd.DataFrame({"Close": closes}, index=pd.to_datetime(dates))
    # add Open/High/Low columns for stop-mode tests
    df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    df["High"] = df[["Open", "Close"]].max(axis=1) * 1.001
    df["Low"] = df[["Open", "Close"]].min(axis=1) * 0.999
    return df


def main():
    df = make_trend()
    sig = MomentumSignal(lookback=20, threshold=0.01)
    sig_df = sig.generate(df.copy())

    # quick checks
    pos_pct = (sig_df['Position'] != 0).mean()
    print(f"Position non-flat fraction: {pos_pct:.2%}")

    trader = PaperTrader(initial_cash=100000.0)
    res = trader.simulate(sig_df, position_col='Position', transaction_cost=0.0)
    print('Trader summary:', trader.summary())
    if hasattr(trader, 'trades'):
        print('Trades count:', len(trader.trades))
        print(trader.trades.head())


if __name__ == '__main__':
    main()
