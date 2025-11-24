"""
Test script for Risk Manager
Demonstrates position sizing methods and risk visualization
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

# Import core modules
from core.risk_manager import RiskManager, RiskConfig
from core.risk_dashboard import RiskDashboard
from core.multi_asset_loader import load_assets
from core.multi_asset_signal import SingleAssetWrapper
from signals.momentum import MomentumSignalV2
from core.portfolio_manager import run_multi_asset_backtest, PortfolioConfig


def test_position_sizing_methods():
    """Test different position sizing methods."""
    print("=" * 80)
    print("TESTING POSITION SIZING METHODS")
    print("=" * 80)
    
    # Load test data
    tickers = ['ES', 'GC']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    print(f"\nLoading data for {tickers}...")
    prices = load_assets(tickers=tickers, start_date=start_date, end_date=end_date)
    
    # Generate signals
    signal_generator = MomentumSignalV2(lookback=120, sma_filter=200)
    multi_signal = SingleAssetWrapper(signal_generator)
    signals = multi_signal.generate(prices)
    
    print(f"✓ Generated signals for {len(signals)} assets\n")
    
    # Test different methods
    methods = ['equal_weight', 'fixed_fraction', 'vol_adjusted']
    
    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"Testing: {method}")
        print('=' * 60)
        
        # Configure risk manager
        risk_config = RiskConfig(
            position_sizing_method=method,
            max_position_size=0.25,
            max_leverage=1.0,
            fixed_fraction=0.05 if method == 'fixed_fraction' else 0.02,
            volatility_target=0.12
        )
        
        risk_mgr = RiskManager(risk_config)
        
        # Test position size calculation
        for ticker in tickers:
            # Calculate volatility from recent returns
            returns = prices[ticker]['Close'].pct_change().dropna()
            vol = risk_mgr.calculate_volatility(ticker, returns)
            
            # Calculate position size
            pos_size = risk_mgr.calculate_position_size(
                ticker=ticker,
                signal=1.0,
                capital=100000,
                positions={},
                volatility=vol
            )
            
            print(f"  {ticker}: Vol={vol:.2%}, Position Size={pos_size:.2%}")
        
        # Run backtest with this method
        config = PortfolioConfig(
            initial_capital=100000,
            rebalance_threshold=0.05,
            transaction_cost_bps=3.0
        )
        
        result, equity_df, trades_df = run_multi_asset_backtest(
            signals_dict=signals,
            prices_dict=prices,
            config=config,
            return_pm=False
        )
        
        metrics = result.calculate_metrics()
        print(f"\n  Results:")
        print(f"    Total Return: {metrics['Total Return']:.2%}")
        print(f"    Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
        print(f"    Max Drawdown: {metrics['Max Drawdown']:.2%}")
        print(f"    Total Trades: {len(trades_df)}")


def test_risk_validation():
    """Test risk validation and limit checks."""
    print("\n" + "=" * 80)
    print("TESTING RISK VALIDATION")
    print("=" * 80 + "\n")
    
    risk_config = RiskConfig(
        max_position_size=0.20,
        max_leverage=1.0,
        correlation_threshold=0.70
    )
    
    risk_mgr = RiskManager(risk_config)
    
    # Test 1: Position size limit
    print("Test 1: Position size validation")
    valid, reason = risk_mgr.validate_trade(
        ticker='ES',
        size=0.15,
        positions={},
        portfolio_value=100000
    )
    print(f"  Size=15%: Valid={valid}, Reason='{reason}'")
    
    valid, reason = risk_mgr.validate_trade(
        ticker='ES',
        size=0.25,
        positions={},
        portfolio_value=100000
    )
    print(f"  Size=25%: Valid={valid}, Reason='{reason}'")
    
    # Test 2: Leverage limit
    print("\nTest 2: Leverage validation")
    positions = {'ES': 50, 'GC': 100}
    prices = {'ES': 4500, 'GC': 2000}
    
    valid, reason = risk_mgr.validate_trade(
        ticker='CL',
        size=0.15,
        positions=positions,
        portfolio_value=100000,
        prices=prices
    )
    print(f"  Adding CL with 50% exposure: Valid={valid}")
    print(f"  Reason: '{reason}'")
    
    # Test 3: Drawdown stop
    print("\nTest 3: Drawdown stop")
    equity = pd.Series([100000, 95000, 90000, 85000, 80000])
    current_dd = (equity.iloc[-1] - equity.max()) / equity.max()
    
    should_stop, reason = risk_mgr.check_stop_conditions(
        current_drawdown=current_dd,
        equity_curve=equity
    )
    print(f"  Drawdown={current_dd:.2%}")
    print(f"  Should Stop={should_stop}, Reason='{reason}'")


def test_risk_dashboard():
    """Test risk dashboard visualization."""
    print("\n" + "=" * 80)
    print("TESTING RISK DASHBOARD")
    print("=" * 80 + "\n")
    
    # Load data
    tickers = ['ES', 'GC']
    prices = load_assets(tickers=tickers, start_date='2020-01-01', end_date='2023-12-31')
    
    # Generate signals
    signal_generator = MomentumSignalV2(lookback=120, sma_filter=200)
    multi_signal = SingleAssetWrapper(signal_generator)
    signals = multi_signal.generate(prices)
    
    # Create risk manager
    risk_config = RiskConfig(
        position_sizing_method='vol_adjusted',
        max_position_size=0.25,
        max_leverage=1.0,
        volatility_target=0.15
    )
    risk_mgr = RiskManager(risk_config)
    
    # Simulate backtest to collect risk metrics
    print("Running backtest with risk tracking...\n")
    
    config = PortfolioConfig(initial_capital=100000)
    result, equity_df, trades_df = run_multi_asset_backtest(
        signals_dict=signals,
        prices_dict=prices,
        config=config,
        return_pm=False
    )
    
    # Simulate risk metrics collection
    # In real integration, this would be done during backtest
    positions = {'ES': 10, 'GC': 15}
    current_prices = {ticker: df['Close'].iloc[-1] for ticker, df in prices.items()}
    
    for i, (date, row) in enumerate(equity_df.iterrows()):
        if i % 20 == 0:  # Sample every 20 days
            portfolio_value = row['TotalValue'] if 'TotalValue' in row else row.get('Equity', 100000)
            
            # Calculate drawdown
            peak = equity_df.iloc[:i+1]['TotalValue'].max() if 'TotalValue' in equity_df.columns else 100000
            current = portfolio_value
            drawdown = (current - peak) / peak if peak > 0 else 0
            
            # Log metrics
            risk_mgr.log_metrics(
                date=pd.Timestamp(date),
                positions=positions,
                prices=current_prices,
                portfolio_value=portfolio_value,
                drawdown=drawdown
            )
    
    # Get metrics dataframe
    risk_metrics_df = risk_mgr.get_metrics_dataframe()
    violations_df = risk_mgr.get_violations_dataframe()
    
    print(f"Collected {len(risk_metrics_df)} risk metric snapshots")
    print(f"Recorded {len(violations_df)} violations")
    
    # Calculate correlation matrix
    returns_data = {}
    for ticker, df in prices.items():
        returns_data[ticker] = df['Close'].pct_change()
    returns_df = pd.DataFrame(returns_data).dropna()
    risk_mgr.update_correlations(returns_df)
    
    # Generate dashboard
    dashboard = RiskDashboard(output_dir='reports')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path('reports') / f'risk_dashboard_{timestamp}.html'
    
    print(f"\nGenerating risk dashboard...")
    dashboard.generate_dashboard(
        risk_metrics_df=risk_metrics_df,
        violations_df=violations_df if not violations_df.empty else None,
        correlation_matrix=risk_mgr.correlation_matrix,
        equity_df=equity_df,
        title="Risk Management Dashboard - Test Run",
        save_path=str(report_path)
    )
    
    print(f"\n✅ Dashboard saved to: {report_path}")
    print(f"\nTo view the dashboard, open the HTML file in your browser.")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RISK MANAGER TEST SUITE")
    print("=" * 80)
    
    try:
        # Run tests
        test_position_sizing_methods()
        test_risk_validation()
        test_risk_dashboard()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
