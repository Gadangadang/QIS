"""
Test script for Reporter module
Generates a sample HTML report from backtest results.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.portfolio_manager import PortfolioConfig, run_multi_asset_backtest
from core.multi_asset_loader import load_assets
from signals.momentum import MomentumSignalV2
from core.reporter import Reporter, quick_report
from datetime import datetime

print("="*60)
print("REPORTER TEST - HTML Report Generation")
print("="*60)

# Load data
print("\n1. Loading data...")
prices = load_assets(['ES', 'GC'], start_date='2020-01-01', end_date='2023-12-31')
print(f"   ✓ Loaded {len(prices)} assets")

# Generate signals
print("\n2. Generating signals...")
signal_gen = MomentumSignalV2(lookback=120)
from core.multi_asset_signal import SingleAssetWrapper
wrapper = SingleAssetWrapper(signal_gen)
signals = wrapper.generate(prices)
print(f"   ✓ Generated signals for {len(signals)} assets")

# Configure portfolio
print("\n3. Configuring portfolio...")
config = PortfolioConfig(
    initial_capital=100000,
    rebalance_threshold=0.05,
    transaction_cost_bps=3.0
)
print(f"   ✓ Initial capital: ${config.initial_capital:,.0f}")

# Run backtest
print("\n4. Running backtest...")
result, equity_df, trades_df = run_multi_asset_backtest(signals, prices, config)
print(f"   ✓ Backtest completed: {len(equity_df)} days, {len(trades_df)} trades")

# Calculate metrics
print("\n5. Calculating metrics...")
metrics = result.calculate_metrics()
print(f"   ✓ {len(metrics)} metrics calculated")

# Display quick console report
quick_report(equity_df, trades_df, metrics)

# Generate HTML report
print("\n" + "="*60)
print("6. Generating HTML Report...")
print("="*60)

reporter = Reporter(output_dir='reports')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
report_path = Path('reports') / f'test_report_{timestamp}.html'

try:
    import plotly
    print(f"   ✓ Plotly version: {plotly.__version__}")
except ImportError:
    print("   ⚠ Plotly not installed - will generate basic report")
    print("   Install with: pip install plotly")

html = reporter.generate_html_report(
    equity_df=equity_df,
    trades_df=trades_df,
    metrics=metrics,
    title="Test Report - ES + GC Portfolio",
    save_path=str(report_path)
)

print("\n" + "="*60)
print("✅ TEST COMPLETED SUCCESSFULLY")
print("="*60)
print(f"\nReport saved to: {report_path.absolute()}")
print("\nTo view the report:")
print(f"  open {report_path.absolute()}")
print("\nOr on macOS:")
print(f"  open -a 'Google Chrome' {report_path.absolute()}")
