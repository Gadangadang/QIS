from pathlib import Path
import pandas as pd
from datetime import date, datetime
import matplotlib.pyplot as plt
from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignal
from signals.ensemble import EnsembleSignal, EnsembleSignalNew
from backtest.backtest_engine import run_train_test
from utils.logger import log_daily_result
from live.paper_trader import PaperTrader

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

def main():
    # ensure logs dir exists
    LOG_DIR.mkdir(exist_ok=True)

    df = load_data()

    # === 1. Define all signals ===
    signals = {
        #"Momentum_60d":  MomentumSignal(lookback=60,  threshold=0.025),
        #"Momentum_120d": MomentumSignal(lookback=120, threshold=0.020),
        #"Momentum_250d": MomentumSignal(lookback=250, threshold=0.018),
        "MR_10":         MeanReversionSignal(window=10,  entry_z=2.2, exit_z=1.0),
        "MR_20":         MeanReversionSignal(window=20,  entry_z=2.0, exit_z=1.0),
        "MR_60":         MeanReversionSignal(window=60,  entry_z=1.8, exit_z=1.0),
        "Ensemble":      EnsembleSignal(),
        "EnsembleNew":   EnsembleSignalNew(),
    }

    # === 2. Run train/test split evaluation (optional sanity check) ===
    print("\n" + "=" * 60)
    print("   QUICK TRAIN/TEST EVALUATION (2015–2025 test period)")
    print("" + "=" * 60)
    for name, signal in signals.items():
        print(f"\n→ {name}")
        result = run_train_test(signal.generate, df.copy())
        summary = result["summary"]
        print(
            f"   Test Sharpe: {summary['sharpe']:+.3f} | Return: {summary['total_return_pct']:+.1f}% | MaxDD: {summary['max_drawdown_pct']:.1f}%"
        )

    # === 3. Run FULL 1990–2025 simulation on Ensemble (the real one) ===
    print("\n" + "=" * 60)
    print("   FULL 1990–2025 BACKTEST — ENSEMBLE STRATEGY")
    print("" + "=" * 60)

    ensemble_signal = signals["EnsembleNew"]
    df_with_signal = ensemble_signal.generate(df.copy())

    trader = PaperTrader(initial_cash=100_000)
    result_df = trader.simulate(df_with_signal, position_col="Position", transaction_cost=3)

    # === 4. Save trades ===
    if hasattr(trader, "trades") and not trader.trades.empty:
        trades_path = LOG_DIR / f"trades_ensemble_{datetime.now().strftime('%Y%m%d')}.csv"
        trader.trades.to_csv(trades_path, index=False)
        print(f"Trades saved → {trades_path.name}")

    # === 5. Print + plot ===
    trader.print_summary()
    trader.plot(title="SPX Ensemble — 1990–2025 Full History", show=True, save=True)
    

    # === 6. Log daily result ===
    log_daily_result(result_df, "SPX Ensemble Strategy")

    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
    



