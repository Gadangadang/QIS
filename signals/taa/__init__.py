"""
TAA (Tactical Asset Allocation) Signal Generators.

Monthly rebalancing signals for multi-asset portfolios using momentum,
carry, value, and macro regime factors.

Available Signals:
- TimeSeriesMomentum: Trend-following based on historical returns
- CrossSectionalMomentum: Relative strength ranking across assets
- YieldCarry: Bond yields and dividend carry signals
- MacroRegime: Economic regime detection signals

Example:
    >>> from signals.taa import TimeSeriesMomentum
    >>> signal = TimeSeriesMomentum(lookback_months=12)
    >>> signals_df = signal.generate(monthly_prices)
"""

from signals.taa.momentum import TimeSeriesMomentum, CrossSectionalMomentum, RiskAdjustedMomentum
from signals.taa.carry import YieldCarry
from signals.taa.ensemble import TAAEnsembleSignal

__all__ = [
    'TimeSeriesMomentum',
    'CrossSectionalMomentum', 
    'RiskAdjustedMomentum',
    'YieldCarry',
    'TAAEnsembleSignal'
]
