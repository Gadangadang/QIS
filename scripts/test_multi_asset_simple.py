"""
Simple multi-asset backtest test.
Tests ES + GC portfolio with momentum signals.
"""
from core.multi_asset_loader import load_assets
from core.multi_asset_signal import SingleAssetWrapper
from signals.momentum import MomentumSignalV2
from core.portfolio_manager import PortfolioManager, PortfolioConfig, run_multi_asset_backtest
import pandas as pd

print("="*60)
print("MULTI-ASSET BACKTEST TEST: ES + GC")
print("="*60)

# Load data
print("\nLoading data...")
prices = load_assets(['ES', 'GC'], start_date='2015-01-01', end_date='2024-12-31')

# Generate signals using existing momentum strategy
print("\nGenerating signals...")
momentum_signal = MomentumSignalV2(lookback=120, entry_threshold=0.02, exit_threshold=-0.01, sma_filter=100)
multi_signal = SingleAssetWrapper(momentum_signal)
signals = multi_signal.generate(prices)

print("\nSignal summary:")
for ticker in ['ES', 'GC']:
    sig_df = signals[ticker]
    n_long = (sig_df['Signal'] == 1).sum()
    n_short = (sig_df['Signal'] == -1).sum()
    n_flat = (sig_df['Signal'] == 0).sum()
    print(f"{ticker}: {n_long} long days, {n_short} short days, {n_flat} flat days")
    print(f"  First 5 dates: {sig_df.index[:5].tolist()}")
    print(f"  First 5 signals: {sig_df['Signal'].head().tolist()}")
    print(f"  First 5 closes: {sig_df['Close'].head().tolist()}")

# Setup portfolio config
config = PortfolioConfig(
    initial_capital=100000,
    rebalance_threshold=0.10,  # 10% drift (higher to reduce rebalancing)
    transaction_cost_bps=3.0
)

# Run backtest
print("\n" + "="*60)
print("Running backtest...")
print("="*60)

pm, equity_curve, trades = run_multi_asset_backtest(signals, prices, config)

# Show results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

metrics = pm.calculate_metrics()
for key, value in metrics.items():
    if isinstance(value, float):
        if 'Return' in key or 'CAGR' in key or 'Volatility' in key or 'Drawdown' in key:
            print(f"{key}: {value:.2%}")
        elif 'Ratio' in key:
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# Trade breakdown
print("\n" + "="*60)
print("TRADES")
print("="*60)

if not trades.empty:
    print(f"\nTotal trades: {len(trades)}")
    print(f"\nBy type:")
    print(trades['Type'].value_counts())
    print(f"\nBy ticker:")
    print(trades['Ticker'].value_counts())
    
    # Show first few trades
    print(f"\nFirst 10 trades:")
    print(trades[['Date', 'Ticker', 'Type', 'Signal', 'Value', 'TransactionCost']].head(10).to_string())
    
    # Transaction cost analysis
    total_tc = trades['TransactionCost'].sum()
    total_traded = trades['Value'].abs().sum()
    print(f"\nTransaction costs: ${total_tc:.2f} ({total_tc/config.initial_capital:.2%} of initial capital)")
    print(f"Total traded volume: ${total_traded:.2f}")
    print(f"TC as % of volume: {total_tc/total_traded:.4%}")
else:
    print("No trades executed!")

# Equity curve
print("\n" + "="*60)
print("EQUITY CURVE SAMPLE")
print("="*60)
print(equity_curve[['Date', 'TotalValue']].head(10).to_string())
print("...")
print(equity_curve[['Date', 'TotalValue']].tail(10).to_string())

print("\n✓ Backtest completed successfully!")
