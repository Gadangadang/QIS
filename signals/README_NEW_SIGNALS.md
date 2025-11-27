# New Signals: Beating SPY Buy-and-Hold

## Overview

Two new signal types designed to outperform SPY buy-and-hold through intelligent market timing and adaptive strategy selection.

## 1. Trend Following Long-Short (`trend_following_long_short.py`)

### Key Features
- **Multi-timeframe confirmation**: Fast (20-day) + Slow (100-day) momentum
- **Volume analysis**: Requires institutional participation (10% above average volume)
- **Volatility regime filter**: Only trades in stable markets (below 70th percentile volatility)
- **Directional flexibility**: Can go LONG, SHORT, or FLAT

### Strategy Logic

**LONG Entry:**
- Fast momentum > 0 (recent uptrend)
- Slow momentum > 2% (strong long-term trend)
- Volume > 1.1x average (institutional confirmation)
- Volatility < 70th percentile (stable market)

**SHORT Entry:**
- Fast momentum < 0 (recent downtrend)
- Slow momentum < -2% (strong bear trend)
- Volume > 1.1x average (conviction in downmove)
- Volatility < 70th percentile (stable bearish trend)

**FLAT (Cash):**
- Conflicting signals
- High volatility (> 70th percentile)
- Low volume (< average)

### Why It Should Beat SPY

1. **Captures bear markets**: Can go short during major downtrends (2020 crash, etc.)
2. **Avoids choppy markets**: Stays flat during high volatility periods
3. **Institutional confirmation**: Only trades when smart money is moving
4. **Risk management**: Multiple filters reduce false signals

### Parameters

```python
TrendFollowingLongShort(
    fast_period=20,          # Short-term trend (default: 20 days)
    slow_period=100,         # Long-term trend (default: 100 days)
    volume_period=50,        # Volume average (default: 50 days)
    momentum_threshold=0.02, # Entry threshold (default: 2%)
    volume_multiplier=1.1,   # Volume requirement (default: 1.1x)
    vol_percentile=0.70      # Max volatility (default: 70th percentile)
)
```

## 2. Adaptive Ensemble (`ensemble.py` - `AdaptiveEnsemble` class)

### Key Features
- **Dynamic strategy weighting**: Adjusts weights based on rolling Sharpe ratios
- **Multi-strategy diversification**: Combines momentum, trend-following, etc.
- **Performance-based allocation**: Increases weight to what's working
- **Signal strength filtering**: Only trades when confidence is high

### Strategy Logic

1. **Generate signals** from all component strategies
2. **Calculate performance** (60-day rolling Sharpe ratio for each)
3. **Update weights** every 20 days based on relative performance
4. **Combine signals** using weighted average
5. **Apply threshold**: Only trade if combined signal > 30% strength
6. **Forward fill**: Stay in position until opposite signal

### Why It Should Beat SPY

1. **Adapts to market regimes**: Weights momentum in trends, mean-reversion in ranges
2. **Diversification**: Reduces single-strategy risk
3. **Performance chasing**: Allocates to strategies that are currently working
4. **Risk control**: Signal threshold prevents weak/conflicting trades
5. **Flexibility**: Can combine any number of strategies

### Parameters

```python
AdaptiveEnsemble(
    strategies=[
        ('momentum', MomentumSignalV2(lookback=60), 0.5),
        ('trend_ls', TrendFollowingLongShort(), 0.5)
    ],
    method='adaptive',          # Use dynamic weighting
    adaptive_lookback=60,       # 60-day performance window
    signal_threshold=0.3,       # 30% min confidence
    rebalance_frequency=20      # Update weights every 20 days
)
```

### Combination Methods

- `'weighted_average'`: Fixed weights (static)
- `'majority_vote'`: Democratic voting (equal weight to all)
- `'unanimous'`: Conservative (only trade when all agree)
- `'adaptive'`: **Recommended** - Dynamic performance-based weighting

## Testing

Use the included test notebook:

```bash
jupyter notebook notebooks/test_new_signals.ipynb
```

### Test Setup
- **Period**: 2015-01-01 to 2024-12-31 (10 years)
- **Asset**: SPY (direct trading, fair comparison)
- **Capital**: $100,000
- **Benchmark**: SPY buy-and-hold
- **Costs**: 3 bps + 2 bps slippage

### Success Metrics

To "beat SPY", we need:

✅ **Higher total return** than buy-and-hold  
✅ **Higher Sharpe ratio** (risk-adjusted returns)  
✅ **Positive alpha** (excess return vs risk)  
✅ **Lower max drawdown** (better risk management)  
✅ **Beta < 1.0** (less volatile than market)

## Implementation Notes

### Transaction Costs
Both strategies include realistic costs:
- 3 bps transaction cost per trade
- 2 bps slippage
- Total: ~5 bps per round-trip

This is conservative for SPY (actual costs often lower).

### Signal Frequency
- **Trend Following LS**: Low turnover (~10-20 trades/year)
- **Adaptive Ensemble**: Medium turnover (~20-40 trades/year)

Lower turnover = Lower costs = Better real-world performance

### Data Requirements

**Minimum data needed:**
- Trend Following LS: ~300 bars (1.5 years daily)
- Adaptive Ensemble: ~400 bars (2 years daily)

**Recommended**: 5+ years for proper evaluation

## Example Usage

```python
# Test Trend Following Long-Short
from signals.trend_following_long_short import TrendFollowingLongShort

signal = TrendFollowingLongShort(
    fast_period=20,
    slow_period=100,
    momentum_threshold=0.02
)

spy_with_signals = signal.generate(spy_data)

# Run backtest
pm = PortfolioManagerV2(initial_capital=100000)
result = pm.run_backtest({'SPY': spy_with_signals}, {'SPY': spy_data})

print(f"Return: {result.metrics['Total Return']:.2%}")
print(f"Sharpe: {result.metrics['Sharpe Ratio']:.2f}")
```

```python
# Test Adaptive Ensemble
from signals.ensemble import AdaptiveEnsemble
from signals.momentum import MomentumSignalV2
from signals.trend_following_long_short import TrendFollowingLongShort

strategies = [
    ('momentum', MomentumSignalV2(lookback=60), 0.5),
    ('trend_ls', TrendFollowingLongShort(), 0.5)
]

ensemble = AdaptiveEnsemble(strategies, method='adaptive')
spy_with_ensemble = ensemble.generate(spy_data)

# Run backtest
result = pm.run_backtest({'SPY': spy_with_ensemble}, {'SPY': spy_data})
```

## Next Steps

1. **Run the test notebook** (`test_new_signals.ipynb`)
2. **Analyze results** - Did we beat SPY?
3. **Parameter optimization** - Can we improve further?
4. **Add to multi-strategy portfolio** - If successful
5. **Walk-forward testing** - Out-of-sample validation
6. **Live paper trading** - Real-time testing

## Parameter Tuning Ideas

If initial results aren't beating SPY:

### Trend Following LS
- **Lower threshold**: `momentum_threshold=0.01` (more trades)
- **Higher threshold**: `momentum_threshold=0.03` (fewer, stronger trades)
- **Relax volume filter**: `volume_multiplier=1.0` (trade more often)
- **Tighter volatility**: `vol_percentile=0.60` (only very stable markets)

### Adaptive Ensemble
- **Longer lookback**: `adaptive_lookback=90` (slower adaptation)
- **Shorter lookback**: `adaptive_lookback=30` (faster adaptation)
- **Lower threshold**: `signal_threshold=0.2` (trade more)
- **More strategies**: Add mean-reversion, volatility signals

## Philosophy

**Key Insight**: You can't beat the market by being more aggressive. You beat it by:
1. **Avoiding big losses** (2008, 2020 crashes)
2. **Staying flat when uncertain** (cash is a position!)
3. **Trading with conviction** (high signal strength only)
4. **Adapting to regimes** (bull/bear/sideways)

The goal isn't to always be in the market. It's to be in the market when you have an **edge**.

---

**Good luck beating SPY! 🚀**
