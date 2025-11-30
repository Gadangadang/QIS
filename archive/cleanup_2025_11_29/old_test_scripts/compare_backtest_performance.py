"""
Compare vectorized vs legacy backtest performance.
"""
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.multi_asset_loader import load_assets
from core.multi_asset_signal import SingleAssetWrapper
from signals.momentum import MomentumSignalV2
from core.portfolio_manager import PortfolioConfig, run_multi_asset_backtest

print("="*60)
print("BACKTEST PERFORMANCE COMPARISON")
print("="*60)

# Load data
print("\nLoading data (ES + GC, 2015-2024)...")
prices = load_assets(['ES', 'GC'], start_date='2015-01-01', end_date='2024-12-31')

# Generate signals
print("Generating signals...")
momentum_signal = MomentumSignalV2(lookback=120, entry_threshold=0.02)
multi_signal = SingleAssetWrapper(momentum_signal)
signals = multi_signal.generate(prices)

# Configure portfolio
config = PortfolioConfig(
    initial_capital=100000,
    rebalance_threshold=0.05,
    transaction_cost_bps=3.0
)

# Test vectorized implementation
print("\n" + "="*60)
print("VECTORIZED BACKTEST (New)")
print("="*60)
start = time.time()
result_vec, equity_vec, trades_vec = run_multi_asset_backtest(
    signals, prices, config, use_vectorized=True
)
time_vec = time.time() - start

print(f"⚡ Execution time: {time_vec:.3f} seconds")
print(f"Total trades: {len(trades_vec)}")
print(f"Final equity: ${equity_vec['TotalValue'].iloc[-1]:,.2f}")

metrics = result_vec.calculate_metrics()
print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
print(f"CAGR: {metrics['CAGR']:.2%}")

# Test legacy implementation
print("\n" + "="*60)
print("LEGACY BACKTEST (Old)")
print("="*60)
start = time.time()
pm_legacy, equity_legacy, trades_legacy = run_multi_asset_backtest(
    signals, prices, config, use_vectorized=False
)
time_legacy = time.time() - start

print(f"🐌 Execution time: {time_legacy:.3f} seconds")
print(f"Total trades: {len(trades_legacy)}")
print(f"Final equity: ${equity_legacy['TotalValue'].iloc[-1]:,.2f}")

metrics_legacy = pm_legacy.calculate_metrics()
print(f"Sharpe Ratio: {metrics_legacy['Sharpe Ratio']:.3f}")
print(f"CAGR: {metrics_legacy['CAGR']:.2%}")

# Compare
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
speedup = time_legacy / time_vec
print(f"Speedup: {speedup:.1f}x faster")
print(f"Time saved: {time_legacy - time_vec:.3f} seconds")

# Verify results match
equity_diff = abs(equity_vec['TotalValue'].iloc[-1] - equity_legacy['TotalValue'].iloc[-1])
print(f"\nFinal equity difference: ${equity_diff:.2f}")
if equity_diff < 1.0:
    print("✅ Results match! Vectorized implementation is correct.")
else:
    print("⚠️  Warning: Results differ. May need adjustment.")

print(f"\n🎯 Target: <5 seconds for 10-year backtest")
if time_vec < 5.0:
    print(f"✅ ACHIEVED! ({time_vec:.3f}s)")
else:
    print(f"❌ Not yet: {time_vec:.3f}s")
