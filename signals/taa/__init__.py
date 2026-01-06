"""
TAA (Tactical Asset Allocation) Signal Generators.

Monthly rebalancing signals for multi-asset portfolios using momentum,
carry, value, and macro regime factors.

Available Signals:
- TimeSeriesMomentum: Trend-following based on historical returns
- CrossSectionalMomentum: Relative strength ranking across assets
- YieldCarry: Bond yields and dividend carry signals
- CAPESignal: Cyclically-adjusted P/E ratio valuation
- RealYieldSpread: Real yield attractiveness
- EarningsYieldSignal: E/P ratio vs bonds
- MacroRegime: Economic regime detection signals

Example:
    >>> from signals.taa import TimeSeriesMomentum, YieldCarry, CAPESignal
    >>> momentum = TimeSeriesMomentum(lookback_months=12)
    >>> carry = YieldCarry(yield_type='spread')
    >>> value = CAPESignal(normalization='zscore')
    >>> signals_df = momentum.generate(monthly_prices)
"""

from signals.taa.momentum import TimeSeriesMomentum, CrossSectionalMomentum, RiskAdjustedMomentum
from signals.taa.carry import YieldCarry, RollYield, CombinedCarry
from signals.taa.value import (
    CAPESignal, 
    RealYieldSpread, 
    EarningsYieldSignal, 
    RelativeValueSignal,
    CommodityValueSignal,
    CrossSectionalValue
)
from signals.taa.ensemble import TAAEnsembleSignal, MultiAssetEnsemble

__all__ = [
    # Momentum
    'TimeSeriesMomentum',
    'CrossSectionalMomentum', 
    'RiskAdjustedMomentum',
    # Carry
    'YieldCarry',
    'RollYield',
    'CombinedCarry',
    # Value
    'CAPESignal',
    'RealYieldSpread',
    'EarningsYieldSignal',
    'RelativeValueSignal',
    'CommodityValueSignal',
    'CrossSectionalValue',
    # Ensemble
    'TAAEnsembleSignal',
    'MultiAssetEnsemble'
]
