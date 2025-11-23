"""
Test Multi-Strategy Signal Framework
Demonstrates using different strategies for different assets.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.multi_asset_loader import load_assets
from core.multi_strategy_signal import MultiStrategySignal, StrategyConfig
from core.portfolio_manager import PortfolioConfig, run_multi_asset_backtest
from signals.momentum import MomentumSignalV2
from signals.mean_reversion import MeanReversionSignal


def test_multi_strategy():
    """Test portfolio with different strategies per asset."""
    
    print("="*70)
    print("MULTI-STRATEGY BACKTEST TEST")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    tickers = ['ES', 'NQ', 'GC']
    prices = load_assets(tickers, start_date='2015-01-01', end_date='2024-12-31')
    print(f"   Loaded {len(tickers)} assets: {tickers}")
    print(f"   Date range: {prices['ES'].index[0]} to {prices['ES'].index[-1]}")
    
    # Method 1: Manual strategy assignment
    print("\n2. Creating multi-strategy configuration (Method 1: Manual)...")
    strategies = {
        'ES': MomentumSignalV2(lookback=120, entry_threshold=0.02, exit_threshold=-0.01, sma_filter=100),
        'NQ': MomentumSignalV2(lookback=120, entry_threshold=0.02, exit_threshold=-0.01, sma_filter=100),
        'GC': MeanReversionSignal(window=50, entry_z=2.0, exit_z=0.5)
    }
    multi_signal_1 = MultiStrategySignal(strategies)
    
    print("   Strategy mapping:")
    for ticker, strategy_name in multi_signal_1.list_strategies().items():
        print(f"     {ticker}: {strategy_name}")
    
    # Method 2: Using StrategyConfig builder
    print("\n3. Creating multi-strategy configuration (Method 2: Builder)...")
    config_builder = StrategyConfig()
    config_builder.add_momentum('ES', lookback=120, entry_threshold=0.02)
    config_builder.add_momentum('NQ', lookback=120, entry_threshold=0.02)
    config_builder.add_mean_reversion('GC', window=50, entry_z=2.0)
    
    print(config_builder.summary())
    multi_signal_2 = config_builder.build()
    
    # Generate signals
    print("\n4. Generating signals...")
    signals = multi_signal_1.generate(prices)
    
    # Analyze signal activity
    print("\n5. Signal Activity Analysis:")
    for ticker in tickers:
        sig = signals[ticker]['Signal']
        n_long = (sig == 1).sum()
        n_flat = (sig == 0).sum()
        pct_long = n_long / len(sig) * 100
        
        strategy_type = type(strategies[ticker]).__name__
        print(f"\n   {ticker} ({strategy_type}):")
        print(f"     Long:  {n_long:4d} days ({pct_long:5.1f}%)")
        print(f"     Flat:  {n_flat:4d} days ({(n_flat/len(sig)*100):5.1f}%)")
    
    # Run backtest
    print("\n6. Running backtest...")
    portfolio_config = PortfolioConfig(
        initial_capital=100000,
        rebalance_threshold=0.10,
        transaction_cost_bps=3.0
    )
    
    pm, equity_curve, trades = run_multi_asset_backtest(signals, prices, portfolio_config)
    metrics = pm.calculate_metrics()
    
    # Display results
    print("\n" + "="*70)
    print("MULTI-STRATEGY PORTFOLIO RESULTS")
    print("="*70)
    print(f"Strategy Configuration:")
    print(f"  ES: Momentum (120-day lookback)")
    print(f"  NQ: Momentum (120-day lookback)")
    print(f"  GC: Mean Reversion (50-day lookback, ±2σ)")
    print(f"\nPerformance Metrics:")
    print(f"  Total Return:       {metrics['Total Return']*100:>7.2f}%")
    print(f"  CAGR:               {metrics['CAGR']*100:>7.2f}%")
    print(f"  Sharpe Ratio:       {metrics['Sharpe Ratio']:>7.3f}")
    print(f"  Max Drawdown:       {metrics['Max Drawdown']*100:>7.2f}%")
    print(f"  Annual Volatility:  {metrics['Annual Volatility']*100:>7.2f}%")
    print(f"  Total Trades:       {metrics['Total Trades']:>7.0f}")
    print(f"  Rebalances:         {metrics['Rebalances']:>7.0f}")
    print(f"  Transaction Costs:  ${metrics['Transaction Costs']:>6.2f}")
    print("="*70)
    
    # Trade summary
    if not trades.empty:
        print(f"\nTrade Summary:")
        print(trades.groupby('Ticker')['Type'].value_counts().unstack(fill_value=0))
    
    print("\n✅ Multi-strategy test complete!")
    return pm, signals, trades, metrics


if __name__ == "__main__":
    try:
        pm, signals, trades, metrics = test_multi_strategy()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
