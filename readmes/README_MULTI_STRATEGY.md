# Multi-Strategy Signal Framework

This framework allows you to apply **different trading strategies to different assets** in a multi-asset portfolio, enabling true strategy diversification.

## Why This Matters

From Week 4 analysis, we discovered:
- **Signal correlation > Return correlation** for diversification
- Using the same momentum strategy on ES, NQ, and GC resulted in highly correlated signals (all long 90%+ of time)
- Gold's -0.04 return correlation with equities didn't help because all strategies were momentum-based

**Solution**: Apply different strategies to different assets:
- **Momentum** for equity indices (ES, NQ) - trend-following
- **Mean Reversion** for commodities (GC) - counter-trend
- **Custom** strategies for specific market conditions

## Quick Start

### Method 1: Manual Strategy Assignment

```python
from core.multi_strategy_signal import MultiStrategySignal
from signals.momentum import MomentumSignalV2
from signals.mean_reversion import MeanReversionSignal

# Define strategy for each asset
strategies = {
    'ES': MomentumSignalV2(lookback=120, entry_threshold=0.02),
    'NQ': MomentumSignalV2(lookback=120, entry_threshold=0.02),
    'GC': MeanReversionSignal(window=50, entry_z=2.0)
}

# Create multi-strategy signal
multi_signal = MultiStrategySignal(strategies)

# Generate signals
signals = multi_signal.generate(prices)
```

### Method 2: Builder Pattern (Recommended)

```python
from core.multi_strategy_signal import StrategyConfig

# Build configuration fluently
config = (StrategyConfig()
          .add_momentum('ES', lookback=120, entry_threshold=0.02)
          .add_momentum('NQ', lookback=120, entry_threshold=0.02)
          .add_mean_reversion('GC', window=50, entry_z=2.0)
          .build())

# Generate signals
signals = config.generate(prices)
```

## Complete Example

```python
from core.multi_asset_loader import load_assets
from core.multi_strategy_signal import StrategyConfig
from core.portfolio_manager import PortfolioConfig, run_multi_asset_backtest

# 1. Load data
prices = load_assets(['ES', 'NQ', 'GC'], start_date='2015-01-01')

# 2. Configure strategies
strategies = (StrategyConfig()
              .add_momentum('ES', lookback=120, entry_threshold=0.02)
              .add_momentum('NQ', lookback=120, entry_threshold=0.02)
              .add_mean_reversion('GC', window=50, entry_z=2.0)
              .build())

# 3. Generate signals
signals = strategies.generate(prices)

# 4. Run backtest
config = PortfolioConfig(
    initial_capital=100000,
    rebalance_threshold=0.10,
    transaction_cost_bps=3.0
)
pm, equity_curve, trades = run_multi_asset_backtest(signals, prices, config)

# 5. Analyze results
metrics = pm.calculate_metrics()
print(f"CAGR: {metrics['CAGR']*100:.2f}%")
print(f"Sharpe: {metrics['Sharpe Ratio']:.3f}")
```

## API Reference

### MultiStrategySignal

Main class for applying different strategies to different assets.

**Methods:**
- `generate(prices)` - Generate signals for all assets
- `add_strategy(ticker, strategy)` - Add/update strategy for asset
- `remove_strategy(ticker)` - Remove strategy for asset
- `get_strategy(ticker)` - Get strategy for asset
- `list_strategies()` - Get dict of {ticker: strategy_name}

### StrategyConfig (Builder)

Fluent interface for building multi-strategy configurations.

**Methods:**
- `add_momentum(ticker, lookback, entry_threshold, exit_threshold, sma_filter)`
- `add_mean_reversion(ticker, window, entry_z, exit_z)`
- `add_custom(ticker, strategy)` - Add any custom strategy instance
- `build()` - Build MultiStrategySignal from configuration
- `summary()` - Get readable summary of configuration

## Strategy Combinations to Explore

### 1. Trend + Counter-Trend
```python
config = (StrategyConfig()
          .add_momentum('ES', lookback=120)      # Trend
          .add_mean_reversion('GC', window=50))  # Counter-trend
```
**Rationale**: Equity indices trend, gold mean-reverts

### 2. Fast + Slow Momentum
```python
config = (StrategyConfig()
          .add_momentum('ES', lookback=60)   # Fast momentum
          .add_momentum('NQ', lookback=180)) # Slow momentum
```
**Rationale**: Different lookbacks capture different trend speeds

### 3. Asset-Specific Optimization
```python
config = (StrategyConfig()
          .add_momentum('ES', lookback=120, entry_threshold=0.02)  # Conservative
          .add_momentum('NQ', lookback=90, entry_threshold=0.03))  # Aggressive
```
**Rationale**: NQ more volatile, requires different thresholds

### 4. Mixed Strategies
```python
from signals.momentum import MomentumSignalV2
from signals.mean_reversion import MeanReversionSignal

# Custom strategy instances with fine-tuned parameters
es_strat = MomentumSignalV2(lookback=120, entry_threshold=0.02, sma_filter=100)
gc_strat = MeanReversionSignal(window=50, entry_z=2.5, exit_z=0.3)

config = (StrategyConfig()
          .add_custom('ES', es_strat)
          .add_custom('GC', gc_strat))
```

## Week 4 vs Week 5 Comparison

### Week 4: Single Strategy (All Momentum)
```
ES: Momentum → Long 94.4% of time
NQ: Momentum → Long 94.4% of time  
GC: Momentum → Long 89.0% of time

Signal Correlation: 0.95+ (highly correlated)
Total Trades: 6 over 25 years
CAGR: ~6%
```

### Week 5: Multi-Strategy
```
ES: Momentum     → Long 94.4% of time
NQ: Momentum     → Long 94.4% of time
GC: Mean Revert → Long 16.3% of time, Flat 54.2%

Signal Correlation: Lower (ES-GC signal correlation reduced)
Total Trades: 311 over 10 years (50x more trading)
GC Trades: 139 (entries/exits/flips)
```

**Key Insight**: Mean reversion on GC provides true diversification - it's flat when equity momentum strategies are long, and trades the opposite pattern.

## Next Steps for Week 5

1. **Optimize GC mean reversion parameters** using walk-forward
2. **Test other strategy combinations**:
   - Volatility breakout for NQ
   - RSI mean reversion for GC
   - Dual momentum (absolute + relative)
3. **Add correlation-based allocation**:
   - Reduce allocation when signal correlation > 0.8
   - Increase allocation when signals diverge
4. **Implement risk budgeting**:
   - Inverse-volatility weighting
   - Target equal risk contribution per asset

## Testing

Run the test script to verify the framework:

```bash
python test_multi_strategy.py
```

Expected output:
- Strategy mapping for each asset
- Signal activity analysis
- Backtest results with performance metrics
- Trade summary by ticker and type

## Files

- `core/multi_strategy_signal.py` - Main framework
- `test_multi_strategy.py` - Example usage and testing
- `signals/momentum.py` - Momentum strategy
- `signals/mean_reversion.py` - Mean reversion strategy
