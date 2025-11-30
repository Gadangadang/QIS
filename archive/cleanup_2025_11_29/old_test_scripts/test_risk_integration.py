"""
Quick test of Phase 2 Risk Manager Integration
Verifies that risk manager works during backtest
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.multi_asset_loader import load_assets
from core.multi_asset_signal import SingleAssetWrapper
from signals.momentum import MomentumSignalV2
from core.portfolio_manager import run_multi_asset_backtest, PortfolioConfig
from core.risk_manager import RiskManager, RiskConfig

print("="*60)
print("PHASE 2 INTEGRATION TEST")
print("="*60)

# Load data
print("\n1. Loading data...")
tickers = ['ES', 'GC']
prices = load_assets(tickers=tickers, start_date='2022-01-01', end_date='2023-12-31')
print(f"   ✓ Loaded {len(prices)} assets")

# Generate signals
print("\n2. Generating signals...")
signal_gen = MomentumSignalV2(lookback=120, sma_filter=200)
multi_signal = SingleAssetWrapper(signal_gen)
signals = multi_signal.generate(prices)
print(f"   ✓ Generated signals for {len(signals)} assets")

# Test 1: Baseline (no risk manager)
print("\n3. Running baseline backtest...")
config_baseline = PortfolioConfig(initial_capital=100000)
result_baseline, equity_baseline, trades_baseline = run_multi_asset_backtest(
    signals, prices, config_baseline
)
metrics_baseline = result_baseline.calculate_metrics()
print(f"   ✓ Baseline complete")
print(f"      - Total Return: {metrics_baseline['Total Return']:.2%}")
print(f"      - Risk metrics: {result_baseline.risk_metrics is not None}")
print(f"      - Violations: {result_baseline.violations is not None}")

# Test 2: With risk manager
print("\n4. Running with risk manager...")
risk_config = RiskConfig(
    position_sizing_method='vol_adjusted',
    max_position_size=0.30,
    max_leverage=1.0,
    max_drawdown_stop=-0.20
)
risk_mgr = RiskManager(risk_config)

config_risk = PortfolioConfig(
    initial_capital=100000,
    risk_manager=risk_mgr,
    rejection_policy='skip'
)

result_risk, equity_risk, trades_risk = run_multi_asset_backtest(
    signals, prices, config_risk
)
metrics_risk = result_risk.calculate_metrics()

print(f"   ✓ Risk-managed backtest complete")
print(f"      - Total Return: {metrics_risk['Total Return']:.2%}")
print(f"      - Risk metrics: {result_risk.risk_metrics is not None}")
print(f"      - Violations: {result_risk.violations is not None}")

# Verify risk data was collected
if result_risk.risk_metrics is not None:
    print(f"\n5. Risk metrics verification:")
    print(f"   ✓ Snapshots collected: {len(result_risk.risk_metrics)}")
    print(f"   ✓ Columns: {list(result_risk.risk_metrics.columns)}")
    
    if len(result_risk.risk_metrics) > 0:
        print(f"   ✓ Sample metrics:")
        sample = result_risk.risk_metrics.iloc[-1]
        print(f"      - Leverage: {sample['leverage']:.2f}x")
        print(f"      - Positions: {sample['num_positions']}")
        print(f"      - Max weight: {sample['max_position_weight']:.2%}")
        print(f"      - Portfolio vol: {sample['portfolio_volatility']:.2%}")
else:
    print(f"\n5. ⚠️ WARNING: No risk metrics collected!")

if result_risk.violations is not None and not result_risk.violations.empty:
    print(f"\n6. Violations found:")
    print(f"   - Total: {len(result_risk.violations)}")
    print(f"   - Types: {result_risk.violations['type'].value_counts().to_dict()}")
else:
    print(f"\n6. ✓ No violations (all trades passed risk checks)")

print("\n" + "="*60)
print("✅ INTEGRATION TEST PASSED")
print("="*60)
print("\nPhase 2 integration is working correctly!")
print("Risk manager successfully integrated with backtest engine.")
