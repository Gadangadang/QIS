"""Quick test: replay the last walk-forward fold with a 10% stop-loss.

Run from repo root:

    python3 -m scripts.test_stop_loss

This will load the dataset, pick the last fold using 60% train / 20% test fractions,
generate momentum signals, run `PaperTrader.simulate(..., stop_loss_pct=0.1)` and
save trades and per-day portfolio values to `logs/walkforward/`.
"""
from pathlib import Path
import pandas as pd

from signals.momentum import MomentumSignal
from core.paper_trader import PaperTrader


def main():
    data_path = Path("Dataset/spx_full_1990_2025.csv")
    out_dir = Path("logs/walkforward")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()

    n = len(df)
    train_size = int(n * 0.6)
    test_size = int(n * 0.2)
    step = test_size

    starts = list(range(0, n - train_size - test_size + 1, step))
    if not starts:
        raise RuntimeError("Not enough data for the requested train/test split")

    # pick the last fold
    start = starts[-1]
    train_end = start + train_size
    test_end = train_end + test_size

    hist_start = max(0, train_end - 250)
    test_with_history = df.iloc[hist_start:test_end].copy()
    test_only = df.iloc[train_end:test_end].copy()

    print(f"Testing fold starting at row {train_end} ({test_only.index[0]}) to {test_only.index[-1]}")

    sig = MomentumSignal(lookback=120, threshold=0.02)
    test_signals = sig.generate(test_with_history)

    trader = PaperTrader(initial_cash=100000.0)
    result_df = trader.simulate(
        test_signals,
        position_col="Position",
        start_date=test_only.index[0],
        end_date=test_only.index[-1],
        transaction_cost=3.0,
        stop_loss_pct=0.1,  # 10% stop loss
    )

    # save outputs
    trades = getattr(trader, "trades", None)
    if trades is not None and not trades.empty:
        trades.to_csv(out_dir / "stop_test_trades.csv", index=False)
        print(f"Saved trades to {out_dir / 'stop_test_trades.csv'}")

    # save per-day portfolio for the test-only window
    pv = result_df.loc[test_only.index, ["PortfolioValue", "Strategy"]].copy()
    pv.to_csv(out_dir / "stop_test_portfolio.csv")
    print(f"Saved portfolio to {out_dir / 'stop_test_portfolio.csv'}")

    print('\n=== Trader Summary ===')
    print(trader.summary())


if __name__ == "__main__":
    main()
