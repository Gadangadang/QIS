#!/usr/bin/env python3
"""Lightweight runner that uses pandas to run the daily pipeline and save outputs.

Place in repository root and run from your own terminal (where pandas is installed).
"""
import argparse
from pathlib import Path
import pandas as pd

from signals.momentum import MomentumSignal
from signals.mean_reversion import MeanReversionSignal
from signals.ensemble import EnsembleSignal, EnsembleSignalNew
from live.paper_trader import PaperTrader


def build_signal(name: str, args: argparse.Namespace):
    n = name.lower()
    if n == "momentum":
        return MomentumSignal(lookback=args.momentum_lookback, threshold=args.threshold)
    if n in ("mean_reversion", "mr"):
        return MeanReversionSignal(window=args.mr_window, entry_z=args.mr_entry_z, exit_z=args.mr_exit_z)
    if n == "ensemble":
        return EnsembleSignal()
    return EnsembleSignalNew()


def main():
    p = argparse.ArgumentParser(description="Run daily pipeline using pandas")
    p.add_argument("--data", default="Dataset/spx_full_1990_2025.csv")
    p.add_argument("--signal", default="ensemble_new", choices=["momentum","mean_reversion","mr","ensemble","ensemble_new"])
    p.add_argument("--momentum-lookback", type=int, default=120)
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--mr-window", type=int, default=20)
    p.add_argument("--mr-entry-z", type=float, default=2.0)
    p.add_argument("--mr-exit-z", type=float, default=1.0)
    p.add_argument("--transaction-cost", type=float, default=3.0)
    p.add_argument("--stop-loss", type=float, default=None)
    p.add_argument("--take-profit", type=float, default=None)
    p.add_argument("--max-hold", type=int, default=None)
    p.add_argument("--stop-mode", choices=["close","low","open"], default="low")
    p.add_argument("--max-pos", type=float, default=0.2)
    p.add_argument("--initial-cash", type=float, default=100000.0)
    p.add_argument("--save-dir", default="logs/daily_test")
    args = p.parse_args()

    df = pd.read_csv(args.data, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["Open","High","Low","Close"]).sort_index()

    sig = build_signal(args.signal, args)
    df_sig = sig.generate(df.copy())

    trader = PaperTrader(initial_cash=args.initial_cash)
    result_df = trader.simulate(
        df_sig,
        position_col="Position",
        transaction_cost=args.transaction_cost,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        max_hold_days=args.max_hold,
        stop_mode=args.stop_mode,
        max_position_pct=args.max_pos,
    )

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(trader, "trades") and not trader.trades.empty:
        trader.trades.to_csv(out / f"trades_daily_{result_df.index[-1].strftime('%Y%m%d')}.csv", index=False)
    # save combined returns and equity
    try:
        result_df["Strategy"].to_csv(out / "combined_returns.csv", header=["Strategy"])
        result_df["PortfolioValue"].to_csv(out / "stitched_equity.csv", header=["PortfolioValue"])
    except Exception:
        pass

    print("Done. Outputs in", out)


if __name__ == "__main__":
    main()
