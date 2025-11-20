import pandas as pd   
from pathlib import Path
from datetime import datetime 
from live.run_daily import load_data
import numpy as np          
import matplotlib.pyplot as plt
from live.paper_trader import PaperTrader
from typing import Callable, Dict, Any, Optional

# ========================= CONFIG =========================
TRAIN_YEARS = 10
TEST_YEARS = 5
INITIAL_TRAIN_END = "2000-01-02"   # First train: 1990 → 1999
DATA_PATH = Path(__file__).parent / "Dataset" / "spx_full_1990_2025.csv"
# =========================================================


def run_walk_forward_fixed(
    signal_class,
    signal_name: str = "Strategy",
    train_years: int = TRAIN_YEARS,
    test_years: int = TEST_YEARS,
    initial_train_end: str = INITIAL_TRAIN_END,
) -> Dict[str, Any]:
    """
    Phase 1: Fixed-parameter walk-forward.
    Returns full results + stitched OOS equity curve.
    """
    full_df = load_data()
    signal = signal_class()

    print(f"\nGenerating signal: {signal_name}")
    full_with_signal = signal.generate(full_df.copy())

    results = []
    equity_segments = []
    train_end = pd.to_datetime(initial_train_end)

    while True:
        train_start = train_end - pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = train_end + pd.DateOffset(years=test_years)

        if test_end > full_df.index.max():
            print("Reached end of data → stopping.\n")
            break

        train_period = f"{train_start.date()} → {train_end.date()}"
        test_period = f"{test_start.date()} → {test_end.date()}"

        print(f"Train: {train_period}")
        print(f" Test: {test_period}")

        test_data = full_df[test_start:test_end].copy()
        test_data["Position"] = full_with_signal.loc[test_data.index, "Position"]

        trader = PaperTrader(initial_cash=100_000)
        result_df = trader.simulate(test_data, position_col="Position", transaction_cost=3)

        summary = trader.summary()
        ret = summary["total_return_pct"]
        sharpe = summary["sharpe"]
        maxdd = summary["max_drawdown_pct"]

        print(f"  OOS → Return: {ret:6.2f}% | Sharpe: {sharpe:+5.2f} | MaxDD: {maxdd:6.2f}%")

        results.append({
            "train_period": train_period,
            "test_period": test_period,
            "return_pct": ret,
            "sharpe": sharpe,
            "maxdd_pct": maxdd,
        })

        # Normalize equity segment
        eq = result_df["PortfolioValue"]
        equity_segments.append(eq / eq.iloc[0])

        train_end += pd.DateOffset(years=test_years)

    # ================= FINAL TRUTH =================
    results_df = pd.DataFrame(results)

    growth_factors = (1 + results_df["return_pct"]/100)
    geo_mean_period = growth_factors.prod() ** (1/len(results_df)) - 1
    cagr = geo_mean_period * 100 / test_years

    print("="*90)
    print(f" WALK-FORWARD RESULTS — {signal_name}")
    print("="*90)
    print(results_df.round(2).to_string(index=False))
    print("="*90)
    print(f"Periods             : {len(results_df)}")
    print(f"Geometric mean ({test_years}y): {geo_mean_period*100:+.2f}% → CAGR ≈ {cagr:+.2f}%")
    print(f"Best {test_years}yr         : {results_df['return_pct'].max():+.1f}%")
    print(f"Worst {test_years}yr        : {results_df['return_pct'].min():+.1f}%")
    print(f"Winning periods     : {(results_df['return_pct'] > 0).sum()}/{len(results_df)}")

    # Stitched OOS equity curve
    oos_equity = pd.concat(equity_segments)
    plt.figure(figsize=(15, 8))
    oos_equity.plot(title=f"Walk-Forward OOS Equity Curve — {signal_name}")
    plt.ylabel("Growth of $100k")
    plt.grid(True, alpha=0.3)
    plt.show()

    return {
        "results_df": results_df,
        "oos_equity": oos_equity,
        "cagr": cagr,
        "geo_mean": geo_mean_period * 100,
    }


if __name__ == "__main__":
    # <<< TEST ANY SIGNAL HERE >>>
    from signals.ensemble import EnsembleSignal
    run_walk_forward_fixed(
        signal_class=EnsembleSignal,
        signal_name="Ensemble_Current_v1",
    )