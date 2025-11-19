"""CLI runner for the walk-forward backtest.

Run from the repository root:

    python3 -m backtest.run_walkforward

Or with arguments, e.g.: 

    python3 -m backtest.run_walkforward --signal momentum --train-frac 0.6 --test-frac 0.2 --lookback 250

This script loads `Dataset/spx_full_1990_2025.csv` by default and calls `run_walk_forward`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from backtest.backtest_engine import run_walk_forward
from signals.momentum import MomentumSignal
from signals.mean_reversion import MeanReversionSignal
from signals.ensemble import EnsembleSignal, EnsembleSignalNew


def build_signal_factory(name: str, args: argparse.Namespace):
    name = name.lower()
    if name == "momentum":
        return lambda: MomentumSignal(lookback=args.momentum_lookback, threshold=args.threshold)
    if name == "mean_reversion" or name == "mr":
        return lambda: MeanReversionSignal(window=args.mr_window, entry_z=args.mr_entry_z, exit_z=args.mr_exit_z)
    if name == "ensemble":
        return lambda: EnsembleSignal()
    if name == "ensemble_new":
        return lambda: EnsembleSignalNew()
    raise ValueError(f"Unknown signal: {name}")


def main():
    p = argparse.ArgumentParser(description="Run anchored walk-forward backtest")
    p.add_argument("--data", default="Dataset/spx_full_1990_2025.csv", help="CSV file with price history")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--train-size", type=int, help="Number of rows for training window")
    grp.add_argument("--train-frac", type=float, default=0.6, help="Fraction of data for training window")
    grp2 = p.add_mutually_exclusive_group()
    grp2.add_argument("--test-size", type=int, help="Number of rows for test window")
    grp2.add_argument("--test-frac", type=float, default=0.2, help="Fraction of data for test window")

    p.add_argument("--lookback", type=int, default=250, help="Lookback history rows to include before each test window")
    p.add_argument("--step", type=int, default=None, help="Step to advance windows; defaults to test_size")
    p.add_argument("--initial-cash", type=float, default=100000.0)
    p.add_argument("--transaction-cost", type=float, default=3.0, help="bps per round-trip (e.g. 3 -> 0.03%)")
    p.add_argument("--signal", default="momentum", choices=["momentum","mean_reversion","mr","ensemble","ensemble_new"], help="Signal to run")

    # signal-specific
    p.add_argument("--momentum-lookback", type=int, default=120)
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--mr-window", type=int, default=20)
    p.add_argument("--mr-entry-z", type=float, default=2.0)
    p.add_argument("--mr-exit-z", type=float, default=1.0)

    p.add_argument("--save-dir", default="logs/walkforward")
    p.add_argument("--stop-loss", type=float, default=None, help="Stop loss as fraction, e.g. 0.1 for 10%")
    p.add_argument("--take-profit", type=float, default=None, help="Take profit as fraction, e.g. 0.1 for 10%")
    p.add_argument("--max-hold", type=int, default=None, help="Max hold days for a trade (integer)")
    p.add_argument("--stop-mode", type=str, default="close", choices=["close","low","open"], help="Stop trigger mode: 'close' (default), 'low' or 'open'")
    p.add_argument("--max-pos", type=float, default=1.0, help="Max position fraction per trade (0-1). Example: 0.2 = 20% of portfolio")

    args = p.parse_args()

    df_path = Path(args.data)
    if not df_path.exists():
        raise FileNotFoundError(f"Data file not found: {df_path}")

    df = pd.read_csv(df_path, index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()

    n = len(df)
    if args.train_size is None:
        train_size = int(n * args.train_frac)
    else:
        train_size = args.train_size
    if args.test_size is None:
        test_size = int(n * args.test_frac)
    else:
        test_size = args.test_size

    print(f"Rows: {n}")
    print(f"Using train_size = {train_size} test_size = {test_size} lookback = {args.lookback}")

    signal_factory = build_signal_factory(args.signal, args)

    res = run_walk_forward(
        signal_factory,
        df,
        train_size=train_size,
        test_size=test_size,
        lookback=args.lookback,
        step=args.step,
        initial_cash=args.initial_cash,
        transaction_cost=args.transaction_cost,
        save_dir=args.save_dir,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        max_hold_days=args.max_hold,
        stop_mode=args.stop_mode,
        max_position_pct=args.max_pos,
    )

    print('\n=== Walk-forward Overall ===')
    print(res["overall"])
    print('\n=== Fold summaries ===')
    for f in res["folds"]:
        print(f)

    # save combined returns
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "combined_returns.csv"
    res["combined_returns"].to_csv(combined_path, header=["Strategy"])
    print(f"Saved combined returns to {combined_path}")


if __name__ == "__main__":
    main()
