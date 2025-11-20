"""Daily runner for paper trading with various signal options."""
from pathlib import Path
import argparse
import pandas as pd
from datetime import date, datetime
import matplotlib.pyplot as plt
from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignal
from signals.ensemble import EnsembleSignal, EnsembleSignalNew
from core.backtest_engine import run_train_test
from utils.logger import log_daily_result
from core.paper_trader import PaperTrader

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOG_DIR = PROJECT_ROOT / "logs"


def load_data() -> pd.DataFrame:
    """Load and fix SPX data — 100% bulletproof."""
    csv_path = "Dataset/spx_full_1990_2025.csv"
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0)

    # NUCLEAR FIX: force correct datetime index
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_index(ascending=True)

    print(f"Loaded {len(df):,} daily bars")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    return df


def main(args: argparse.Namespace | None = None):
    # ensure logs dir exists
    LOG_DIR.mkdir(exist_ok=True)

    # If args not provided, parse CLI
    if args is None:
        p = argparse.ArgumentParser(description="Run daily backtest/runner with configurable params")
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
        p.add_argument("--stop-mode", type=str, default="close", choices=["close","low","open"])
        p.add_argument("--max-pos", type=float, default=1.0)
        p.add_argument("--save-dir", default=str(LOG_DIR))
        args = p.parse_args()

    df = load_data() if getattr(args, 'data', None) is None else pd.read_csv(args.data, index_col=0)
    if isinstance(df, pd.DataFrame):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()

    # === 1. Build signal based on args ===
    sig_name = args.signal.lower() if hasattr(args, 'signal') else 'ensemble_new'
    if sig_name == 'momentum':
        sig = MomentumSignal(lookback=getattr(args, 'momentum_lookback', 120), threshold=getattr(args, 'threshold', 0.02))
    elif sig_name in ('mean_reversion', 'mr'):
        sig = MeanReversionSignal(window=getattr(args, 'mr_window', 20), entry_z=getattr(args, 'mr_entry_z', 2.0), exit_z=getattr(args, 'mr_exit_z', 1.0))
    elif sig_name == 'ensemble':
        sig = EnsembleSignal()
    else:
        sig = EnsembleSignalNew()

    # Optional quick train/test evaluation
    print("\n" + "=" * 60)
    print("   QUICK TRAIN/TEST EVALUATION (sanity check)")
    print("" + "=" * 60)
    try:
        res = run_train_test(sig.generate, df.copy())
        print('Train/Test summary:', res['summary'])
    except Exception as e:
        print('Train/test failed:', e)

    # === 2. Run full historical simulation ===
    df_with_signal = sig.generate(df.copy())

    trader = PaperTrader(initial_cash=getattr(args, 'initial_cash', 100000.0))
    result_df = trader.simulate(
        df_with_signal,
        position_col="Position",
        transaction_cost=getattr(args, 'transaction_cost', 3.0),
        stop_loss_pct=getattr(args, 'stop_loss', None),
        take_profit_pct=getattr(args, 'take_profit', None),
        max_hold_days=getattr(args, 'max_hold', None),
        stop_mode=getattr(args, 'stop_mode', 'close'),
        max_position_pct=getattr(args, 'max_pos', 1.0),
    )

    # === 3. Save outputs to save_dir ===
    save_dir = Path(getattr(args, 'save_dir', LOG_DIR))
    save_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(trader, "trades") and not trader.trades.empty:
        trader.trades.to_csv(save_dir / f"trades_daily_{datetime.now().strftime('%Y%m%d')}.csv", index=False)

    # save combined returns and stitched equity
    try:
        result_df['Strategy'].to_csv(save_dir / 'combined_returns.csv', header=['Strategy'])
        result_df['PortfolioValue'].to_csv(save_dir / 'stitched_equity.csv', header=['PortfolioValue'])
    except Exception:
        pass

    # === 4. Print + plot ===
    trader.print_summary()
    try:
        trader.plot(title="SPX Strategy — Full History", show=True, save=True)
    except Exception:
        pass

    # === 5. Log daily result ===
    try:
        log_daily_result(result_df, "SPX Strategy")
    except Exception:
        pass

    print("\nDaily run complete. Outputs saved to:", save_dir)


if __name__ == "__main__":
    main()
