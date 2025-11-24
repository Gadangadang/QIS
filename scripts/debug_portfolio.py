"""
"""<br/>Debug portfolio manager with simple manual test.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.portfolio_manager import PortfolioManager, PortfolioConfig
import pandas as pd

config = PortfolioConfig(initial_capital=100000, rebalance_threshold=0.05, transaction_cost_bps=3.0)
pm = PortfolioManager(config)

# Day 1: Initialize with both ES and GC long
prices = {'ES': 5000.0, 'GC': 2000.0}
signals = {'ES': 1, 'GC': 1}

print("Day 1: Initialize")
print(f"Prices: {prices}")
print(f"Signals: {signals}")

pm.initialize_positions(prices, signals)

print(f"\nPortfolio value: ${pm.portfolio_value:,.2f}")
print(f"Cash: ${pm.cash:,.2f}")
print(f"Positions:")
for ticker, pos in pm.positions.items():
    print(f"  {ticker}: {pos['shares']:.2f} shares @ ${pos['current_price']:.2f} = ${pos['value']:,.2f} ({pos['weight']:.1%})")

# Day 2: Prices change
print("\n" + "="*60)
print("Day 2: Prices change")
prices = {'ES': 5100.0, 'GC': 1950.0}
signals = {'ES': 1, 'GC': 1}

print(f"Prices: {prices}")
pm.update_positions(prices)

print(f"\nPortfolio value: ${pm.portfolio_value:,.2f}")
print(f"Cash: ${pm.cash:,.2f}")
print(f"Positions:")
for ticker, pos in pm.positions.items():
    print(f"  {ticker}: {pos['shares']:.2f} shares @ ${pos['current_price']:.2f} = ${pos['value']:,.2f} ({pos['weight']:.1%})")

# Check if rebalance needed
needs_rebalance = pm.check_rebalance_needed(signals)
print(f"\nNeeds rebalance: {needs_rebalance}")

if needs_rebalance:
    print("\nRebalancing...")
    date = pd.Timestamp('2020-01-02')
    pm.rebalance(prices, signals, date)
    pm.update_positions(prices)
    
    print(f"\nAfter rebalance:")
    print(f"Portfolio value: ${pm.portfolio_value:,.2f}")
    print(f"Cash: ${pm.cash:,.2f}")
    print(f"Positions:")
    for ticker, pos in pm.positions.items():
        print(f"  {ticker}: {pos['shares']:.2f} shares @ ${pos['current_price']:.2f} = ${pos['value']:,.2f} ({pos['weight']:.1%})")

# Day 3: Another price change
print("\n" + "="*60)
print("Day 3: Another price change")
prices = {'ES': 5200.0, 'GC': 2000.0}
signals = {'ES': 1, 'GC': 1}

print(f"Prices: {prices}")
pm.update_positions(prices)

print(f"\nPortfolio value: ${pm.portfolio_value:,.2f}")
print(f"Cash: ${pm.cash:,.2f}")
print(f"Positions:")
for ticker, pos in pm.positions.items():
    print(f"  {ticker}: {pos['shares']:.2f} shares @ ${pos['current_price']:.2f} = ${pos['value']:,.2f} ({pos['weight']:.1%})")
