"""Unified backtest CLI.

Usage examples (from repo root):

  # Run anchored walk-forward then diagnostics
  python3 -m backtest.runner walkforward --signal momentum --train-frac 0.6 --test-frac 0.2 \
      --lookback 250 --save-dir logs/walkforward_expt --stop-loss 0.1 --stop-mode low --max-pos 0.2

  # Run a full historical backtest (single run) and diagnostics
  python3 -m backtest.runner backtest --signal momentum --save-dir logs/full_backtest

  # Run the daily runner (interactive plotting/logging)
  python3 -m backtest.runner daily

This script centralizes outputs so diagnostics and analysis are always saved
under the chosen `--save-dir`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json

import pandas as pd

from backtest.backtest_engine import run_walk_forward, run_train_test
from live.paper_trader import PaperTrader
from backtest import run_diagnostics as diag_module
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


def write_combined_returns(series: pd.Series, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # series should be indexed by date
    series.to_csv(out_path, header=["Strategy"])


def run_walkforward_cmd(args: argparse.Namespace):
    df = pd.read_csv(args.data, index_col=0)
    df.index = pd.to_datetime(df.index)
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

    # save combined returns
    combined_path = Path(args.save_dir) / "combined_returns.csv"
    write_combined_returns(res["combined_returns"], combined_path)

    # save stitched equity plot and combined returns equity plot
    try:
        stitched = res["stitched_equity"]
        out_dir = Path(args.save_dir)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 5))
        plt.plot(stitched.index, stitched.values, label="Stitched Portfolio")
        plt.title("Stitched Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig.savefig(out_dir / "stitched_equity.png")
        plt.close(fig)

        # combined returns -> cumulative equity
        comb = res["combined_returns"].fillna(0)
        cum = (1 + comb).cumprod() * args.initial_cash
        fig2 = plt.figure(figsize=(10, 5))
        plt.plot(cum.index, cum.values, label="Combined Strategy Equity")
        plt.title("Combined Strategy Equity (OOS periods)")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig2.savefig(out_dir / "combined_returns_equity.png")
        plt.close(fig2)
    except Exception:
        pass

    # print fold-level summaries
    try:
        print('\nFold summaries:')
        for f in res.get('folds', []):
            print(
                f" Fold {f['fold']}: {f['start'].date() if f['start'] is not None else 'NA'} -> {f['end'].date() if f['end'] is not None else 'NA'} | Return: {f['fold_return_pct']:+.2%} | Sharpe: {f['sharpe']:+.3f} | MaxDD: {f['max_drawdown']:.2%} | Trades: {f['n_trades']}"
            )
        overall = res.get('overall', {})
        print('\nOverall:')
        print(f" Total return: {overall.get('total_return_pct', 0):+.2%} | Sharpe: {overall.get('sharpe', 0):+.3f} | MaxDD: {overall.get('max_drawdown', 0):.2%} | Folds: {overall.get('n_folds', 0)}")
    except Exception:
        pass

    # run diagnostics on this save_dir
    run_diagnostics_on_dir(args.save_dir, args.data)

    print('\nWalk-forward complete. Outputs in', args.save_dir)


def run_full_backtest_cmd(args: argparse.Namespace):
    df = pd.read_csv(args.data, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["Close"]).sort_index()

    sig_factory = build_signal_factory(args.signal, args)
    sig = sig_factory() if callable(sig_factory) else sig_factory
    df_with_signal = sig.generate(df.copy())

    trader = PaperTrader(initial_cash=args.initial_cash)
    result_df = trader.simulate(
        df_with_signal,
        position_col="Position",
        transaction_cost=args.transaction_cost,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        max_hold_days=args.max_hold,
        stop_mode=args.stop_mode,
        max_position_pct=args.max_pos,
    )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save trades
    if hasattr(trader, "trades") and not trader.trades.empty:
        trader.trades.to_csv(out_dir / "trades_full.csv", index=False)

    # save combined returns (Strategy series)
    write_combined_returns(result_df["Strategy"], out_dir / "combined_returns.csv")

    # save stitched equity-like series (PortfolioValue)
    result_df["PortfolioValue"].to_csv(out_dir / "stitched_equity.csv", header=["PortfolioValue"])    

    # print and save summary + equity plot
    try:
        trader.print_summary()
    except Exception:
        pass

    try:
        # save equity plot into out_dir
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result_df.index, result_df['PortfolioValue'], label='Portfolio')
        ax.set_title('Full Backtest Equity Curve')
        ax.set_ylabel('Portfolio Value')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(out_dir / 'equity_curve.png')
        plt.close(fig)
    except Exception:
        pass

    # run diagnostics
    run_diagnostics_on_dir(str(out_dir), args.data)

    print('\nFull backtest complete. Outputs in', out_dir)


def run_daily_cmd(args: argparse.Namespace):
    # delegate to live.run_daily.main for interactive daily runner
    import live.run_daily as rd

    # rd.main(args) will parse and run using the provided Namespace
    try:
        rd.main(args)
    except TypeError:
        # fallback if rd.main doesn't accept args (backwards compat)
        rd.main()

    # after daily runner completes, try to load saved stitched_equity/combined_returns and plot/print summary
    out_dir = Path(args.save_dir)
    try:
        # print simple summary if stitched_equity exists
        se = pd.read_csv(out_dir / 'stitched_equity.csv', index_col=0, parse_dates=True)
        cr = pd.read_csv(out_dir / 'combined_returns.csv', index_col=0, parse_dates=True)
        # print simple summary
        start_val = float(se.iloc[0,0])
        end_val = float(se.iloc[-1,0])
        total_ret = end_val / start_val - 1
        print(f"\nDaily run summary: Start {start_val:.2f} End {end_val:.2f} Total return {total_ret:+.2%}")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(se.index, se.iloc[:,0].values, label='Stitched Equity')
            ax.set_title('Daily Run Stitched Equity')
            ax.set_ylabel('Portfolio Value')
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.savefig(out_dir / 'daily_stitched_equity.png')
            plt.close(fig)
        except Exception:
            pass
    except Exception:
        print('No stitched_equity/combined_returns found in', out_dir)

    # run diagnostics on save_dir so results are collected like other commands
    try:
        run_diagnostics_on_dir(args.save_dir, args.data)
    except Exception:
        print('Diagnostics failed for daily run (check paths):', args.save_dir)


def run_testfold_cmd(args: argparse.Namespace):
    # Replay the last fold (fast) with provided parameters and save outputs
    df = pd.read_csv(args.data, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["Close"]).sort_index()

    n = len(df)
    train_size = int(n * args.train_frac)
    test_size = int(n * args.test_frac)
    step = test_size
    starts = list(range(0, n - train_size - test_size + 1, step))
    if not starts:
        raise RuntimeError("Not enough data for the requested train/test split")

    start = starts[-1]
    train_end = start + train_size
    test_end = train_end + test_size
    hist_start = max(0, train_end - args.lookback)
    test_with_history = df.iloc[hist_start:test_end].copy()
    test_only = df.iloc[train_end:test_end].copy()

    signal_factory = build_signal_factory(args.signal, args)
    sig = signal_factory() if callable(signal_factory) else signal_factory
    test_signals = sig.generate(test_with_history)

    trader = PaperTrader(initial_cash=args.initial_cash)
    result_df = trader.simulate(
        test_signals,
        position_col="Position",
        start_date=test_only.index[0],
        end_date=test_only.index[-1],
        transaction_cost=args.transaction_cost,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        max_hold_days=args.max_hold,
        stop_mode=args.stop_mode,
        max_position_pct=args.max_pos,
    )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(trader, "trades") and not trader.trades.empty:
        trader.trades.to_csv(out_dir / "stop_test_trades.csv", index=False)
    # save per-day portfolio for the test-only window
    pv = result_df.loc[test_only.index, ["PortfolioValue", "Strategy"]].copy()
    pv.to_csv(out_dir / "stop_test_portfolio.csv")
    # save combined returns for the test-only period
    write_combined_returns(result_df.loc[test_only.index, "Strategy"], out_dir / "combined_returns.csv")

    # run diagnostics on this output folder
    run_diagnostics_on_dir(out_dir, args.data)

    print('\nTest-fold complete. Outputs in', out_dir)


def run_sweep_cmd(args: argparse.Namespace):
    # Small grid sweep on last fold for stop-loss, stop-mode, max-pos
    from itertools import product

    stop_losses = [0.05, 0.1, 0.15]
    stop_modes = ["close", "low", "open"]
    max_poses = [0.1, 0.2, 0.5, 1.0]

    df = pd.read_csv(args.data, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["Close"]).sort_index()

    n = len(df)
    train_size = int(n * args.train_frac)
    test_size = int(n * args.test_frac)
    step = test_size
    starts = list(range(0, n - train_size - test_size + 1, step))
    if not starts:
        raise RuntimeError("Not enough data for the requested train/test split")

    start = starts[-1]
    train_end = start + train_size
    test_end = train_end + test_size
    hist_start = max(0, train_end - args.lookback)
    test_with_history = df.iloc[hist_start:test_end].copy()
    test_only = df.iloc[train_end:test_end].copy()

    signal_factory = build_signal_factory(args.signal, args)
    sig = signal_factory() if callable(signal_factory) else signal_factory
    test_signals = sig.generate(test_with_history)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows = []
    for sl, sm, mp in product(stop_losses, stop_modes, max_poses):
        trader = PaperTrader(initial_cash=args.initial_cash)
        res = trader.simulate(
            test_signals,
            position_col="Position",
            start_date=test_only.index[0],
            end_date=test_only.index[-1],
            transaction_cost=args.transaction_cost,
            stop_loss_pct=sl,
            stop_mode=sm,
            max_position_pct=mp,
        )
        # compute summary
        pv = res.loc[test_only.index, "PortfolioValue"]
        fold_return = float(pv.iloc[-1] / pv.iloc[0] - 1) if len(pv) > 0 else 0.0
        sweep_rows.append({"stop_loss": sl, "stop_mode": sm, "max_pos": mp, "fold_return": fold_return})

    # save sweep results
    import csv
    with (out_dir / "sweep_results.csv").open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=["stop_loss", "stop_mode", "max_pos", "fold_return"])
        w.writeheader()
        for r in sweep_rows:
            w.writerow(r)

    print('Sweep complete. Results saved to', out_dir / 'sweep_results.csv')


def run_diagnostics_on_dir(save_dir: str | Path, data_path: str):
    save_dir = Path(save_dir)
    # prepare argv for run_diagnostics.main
    argv = [
        "run_diagnostics",
        "--combined",
        str(save_dir / "combined_returns.csv"),
        "--data",
        str(data_path),
        "--trades-glob",
        str(save_dir / "trades*.csv"),
        "--stitched",
        str(save_dir / "stitched_equity.csv"),
        "--out",
        str(save_dir / "diagnostics.txt"),
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        diag_module.main()
    finally:
        sys.argv = old_argv


def main():
    p = argparse.ArgumentParser(description="Unified backtest runner")
    sub = p.add_subparsers(dest="cmd")

    # common args for backtests
    def add_common(parser):
        parser.add_argument("--data", default="Dataset/spx_full_1990_2025.csv")
        parser.add_argument("--initial-cash", type=float, default=100000.0)
        parser.add_argument("--transaction-cost", type=float, default=3.0)
        parser.add_argument("--signal", default="momentum", choices=["momentum","mean_reversion","mr","ensemble","ensemble_new"])
        parser.add_argument("--momentum-lookback", type=int, default=120)
        parser.add_argument("--threshold", type=float, default=0.02)
        parser.add_argument("--mr-window", type=int, default=20)
        parser.add_argument("--mr-entry-z", type=float, default=2.0)
        parser.add_argument("--mr-exit-z", type=float, default=1.0)
        parser.add_argument("--save-dir", default="logs/walkforward")
        parser.add_argument("--stop-loss", type=float, default=None)
        parser.add_argument("--take-profit", type=float, default=None)
        parser.add_argument("--max-hold", type=int, default=None)
        parser.add_argument("--stop-mode", type=str, default="close", choices=["close","low","open"])
        parser.add_argument("--max-pos", type=float, default=1.0)

    p_wf = sub.add_parser("walkforward", help="Run anchored walk-forward")
    add_common(p_wf)
    p_wf.add_argument("--train-size", type=int)
    p_wf.add_argument("--train-frac", type=float, default=0.6)
    p_wf.add_argument("--test-size", type=int)
    p_wf.add_argument("--test-frac", type=float, default=0.2)
    p_wf.add_argument("--lookback", type=int, default=250)
    p_wf.add_argument("--step", type=int, default=None)

    p_bt = sub.add_parser("backtest", help="Run full historical backtest")
    add_common(p_bt)

    p_test = sub.add_parser("testfold", help="Run the last walk-forward fold quickly with custom params")
    add_common(p_test)
    p_test.add_argument("--train-frac", type=float, default=0.6)
    p_test.add_argument("--test-frac", type=float, default=0.2)
    p_test.add_argument("--lookback", type=int, default=250)

    p_sweep = sub.add_parser("sweep", help="(Optional) run a small parameter sweep on the last fold and save results")
    add_common(p_sweep)
    p_sweep.add_argument("--train-frac", type=float, default=0.6)
    p_sweep.add_argument("--test-frac", type=float, default=0.2)
    p_sweep.add_argument("--lookback", type=int, default=250)

    p_daily = sub.add_parser("daily", help="Run the interactive daily runner")
    add_common(p_daily)

    p_diag = sub.add_parser("diagnostics", help="Run diagnostics on an existing save-dir")
    p_diag.add_argument("--save-dir", default="logs/walkforward")
    p_diag.add_argument("--data", default="Dataset/spx_full_1990_2025.csv")

    args = p.parse_args()
    if args.cmd == "walkforward":
        run_walkforward_cmd(args)
    elif args.cmd == "backtest":
        run_full_backtest_cmd(args)
    elif args.cmd == "testfold":
        run_testfold_cmd(args)
    elif args.cmd == "sweep":
        run_sweep_cmd(args)
    elif args.cmd == "daily":
        run_daily_cmd(args)
    elif args.cmd == "diagnostics":
        run_diagnostics_on_dir(args.save_dir, args.data)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
