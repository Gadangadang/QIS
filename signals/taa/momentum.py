"""
TAA Momentum Signals for Monthly Asset Allocation.

All signals return expected returns or z-scores suitable for portfolio optimization,
not binary long/short positions like daily trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from signals.base import SignalModel


class TimeSeriesMomentum(SignalModel):
    """
    Time-Series Momentum (TSM) signal for TAA.
    
    Calculates momentum using multiple lookback periods and optional
    volatility adjustment. Returns expected return forecasts rather than
    binary signals.
    
    Args:
        lookback_months: Primary lookback period in months (default: 12)
        additional_lookbacks: Additional lookback periods for ensemble (default: [3, 6])
        vol_adjust: Volatility-adjust returns for risk parity (default: True)
        vol_window: Window for volatility calculation in months (default: 36)
        min_periods: Minimum periods required for signal (default: 12)
    
    Returns:
        DataFrame with columns:
        - Signal: Expected return forecast (continuous, not binary)
        - Momentum_12M: 12-month momentum
        - Momentum_6M: 6-month momentum (if enabled)
        - Momentum_3M: 3-month momentum (if enabled)
        - Volatility: Annualized volatility
        - RiskAdjustedMomentum: Sharpe-like momentum score
    
    Example:
        >>> tsm = TimeSeriesMomentum(lookback_months=12, vol_adjust=True)
        >>> signals = tsm.generate(monthly_prices)
        >>> # Use signals['Signal'] as expected return input to optimizer
    """
    
    def __init__(
        self,
        lookback_months: int = 12,
        additional_lookbacks: List[int] = [3, 6],
        vol_adjust: bool = True,
        vol_window: int = 36,
        min_periods: int = 12
    ):
        if lookback_months <= 0:
            raise ValueError(f"lookback_months must be positive, got {lookback_months}")
        if vol_window < lookback_months:
            raise ValueError(f"vol_window ({vol_window}) should be >= lookback_months ({lookback_months})")
        
        self.lookback_months = lookback_months
        self.additional_lookbacks = additional_lookbacks or []
        self.vol_adjust = vol_adjust
        self.vol_window = vol_window
        self.min_periods = min_periods
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate time-series momentum signals.
        
        Args:
            df: Monthly price DataFrame with 'Close' column
        
        Returns:
            DataFrame with Signal column containing expected return forecasts
        """
        df = df.copy()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Primary momentum (12-month)
        df[f'Momentum_{self.lookback_months}M'] = (
            df['Close'] / df['Close'].shift(self.lookback_months) - 1
        )
        
        # Additional lookback periods
        for lookback in self.additional_lookbacks:
            df[f'Momentum_{lookback}M'] = (
                df['Close'] / df['Close'].shift(lookback) - 1
            )
        
        # Volatility calculation
        df['Volatility'] = df['Returns'].rolling(
            window=self.vol_window, 
            min_periods=self.min_periods
        ).std() * np.sqrt(12)  # Annualize monthly vol
        
        # Risk-adjusted momentum (Sharpe-like)
        if self.vol_adjust:
            df['RiskAdjustedMomentum'] = (
                df[f'Momentum_{self.lookback_months}M'] / df['Volatility']
            )
            # Use risk-adjusted as primary signal
            df['Signal'] = df['RiskAdjustedMomentum']
        else:
            # Use raw momentum as signal
            df['Signal'] = df[f'Momentum_{self.lookback_months}M']
        
        # Ensemble: Average multiple lookbacks if provided
        if self.additional_lookbacks:
            momentum_cols = [f'Momentum_{lb}M' for lb in [self.lookback_months] + self.additional_lookbacks]
            if self.vol_adjust:
                # Vol-adjust each momentum before averaging
                for col in momentum_cols:
                    df[f'{col}_VolAdj'] = df[col] / df['Volatility']
                avg_cols = [f'{col}_VolAdj' for col in momentum_cols]
            else:
                avg_cols = momentum_cols
            
            df['Signal'] = df[avg_cols].mean(axis=1)
        
        # Set warm-up period to NaN (not 0, since 0 is a valid signal)
        warmup_periods = max(self.lookback_months, self.vol_window)
        df.iloc[:warmup_periods, df.columns.get_loc('Signal')] = np.nan
        
        return df


class CrossSectionalMomentum(SignalModel):
    """
    Cross-Sectional Momentum (CSM) for TAA.
    
    Ranks assets by momentum and returns z-scores for portfolio optimization.
    Requires multiple assets to compute cross-sectional statistics.
    
    Args:
        lookback_months: Lookback period for momentum (default: 12)
        normalization: Method for normalization ('zscore', 'rank', 'minmax') (default: 'zscore')
        vol_adjust: Volatility-adjust before ranking (default: True)
    
    Returns:
        DataFrame with 'Signal' column containing z-scores or normalized ranks
    
    Usage:
        This signal generator should be applied to multiple assets simultaneously:
        >>> csm = CrossSectionalMomentum(lookback_months=12)
        >>> # Apply to dict of DataFrames
        >>> signals = {ticker: csm.generate(df) for ticker, df in prices.items()}
        >>> # Then normalize cross-sectionally
        >>> all_signals = pd.DataFrame({
        ...     ticker: sig['Momentum_12M'] 
        ...     for ticker, sig in signals.items()
        ... })
        >>> zscore_signals = (all_signals - all_signals.mean(axis=1, keepdims=True)) / all_signals.std(axis=1, keepdims=True)
    
    Note:
        For proper cross-sectional signals, use TAAEnsembleSignal which handles
        multiple assets correctly.
    """
    
    def __init__(
        self,
        lookback_months: int = 12,
        normalization: str = 'zscore',
        vol_adjust: bool = True
    ):
        if normalization not in ['zscore', 'rank', 'minmax']:
            raise ValueError(f"normalization must be 'zscore', 'rank', or 'minmax', got {normalization}")
        
        self.lookback_months = lookback_months
        self.normalization = normalization
        self.vol_adjust = vol_adjust
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cross-sectional momentum for single asset.
        
        Note: For true cross-sectional signals, normalization happens
        across all assets. This method just calculates raw momentum.
        Use TAAEnsembleSignal for proper cross-sectional normalization.
        """
        df = df.copy()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Momentum
        df[f'Momentum_{self.lookback_months}M'] = (
            df['Close'] / df['Close'].shift(self.lookback_months) - 1
        )
        
        # Volatility if needed
        if self.vol_adjust:
            df['Volatility'] = df['Returns'].rolling(
                window=self.lookback_months * 2,
                min_periods=self.lookback_months
            ).std() * np.sqrt(12)
            
            df['Signal'] = df[f'Momentum_{self.lookback_months}M'] / df['Volatility']
        else:
            df['Signal'] = df[f'Momentum_{self.lookback_months}M']
        
        # Warm-up
        df.iloc[:self.lookback_months, df.columns.get_loc('Signal')] = np.nan
        
        return df


class RiskAdjustedMomentum(SignalModel):
    """
    Risk-Adjusted Momentum using Sharpe or Sortino ratios.
    
    Args:
        lookback_months: Lookback period (default: 12)
        risk_metric: 'sharpe' or 'sortino' (default: 'sharpe')
        risk_free_rate: Annual risk-free rate (default: 0.02)
        min_periods: Minimum periods for calculation (default: 12)
    
    Returns:
        DataFrame with 'Signal' column containing Sharpe/Sortino ratios
    
    Example:
        >>> ram = RiskAdjustedMomentum(lookback_months=12, risk_metric='sharpe')
        >>> signals = ram.generate(monthly_prices)
    """
    
    def __init__(
        self,
        lookback_months: int = 12,
        risk_metric: str = 'sharpe',
        risk_free_rate: float = 0.02,
        min_periods: int = 12
    ):
        if risk_metric not in ['sharpe', 'sortino']:
            raise ValueError(f"risk_metric must be 'sharpe' or 'sortino', got {risk_metric}")
        
        self.lookback_months = lookback_months
        self.risk_metric = risk_metric
        self.risk_free_rate = risk_free_rate
        self.min_periods = min_periods
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate risk-adjusted momentum signals."""
        df = df.copy()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Excess returns (monthly)
        monthly_rf = (1 + self.risk_free_rate) ** (1/12) - 1
        df['ExcessReturns'] = df['Returns'] - monthly_rf
        
        # Rolling metrics
        rolling_mean = df['ExcessReturns'].rolling(
            window=self.lookback_months,
            min_periods=self.min_periods
        ).mean()
        
        if self.risk_metric == 'sharpe':
            # Standard deviation of all returns
            rolling_risk = df['Returns'].rolling(
                window=self.lookback_months,
                min_periods=self.min_periods
            ).std()
        else:  # sortino
            # Downside deviation (only negative returns)
            def downside_std(x):
                downside = x[x < 0]
                return downside.std() if len(downside) > 0 else np.nan
            
            rolling_risk = df['Returns'].rolling(
                window=self.lookback_months,
                min_periods=self.min_periods
            ).apply(downside_std, raw=False)
        
        # Sharpe/Sortino ratio (annualized)
        df[f'{self.risk_metric.capitalize()}Ratio'] = (
            rolling_mean / rolling_risk * np.sqrt(12)
        )
        
        # Use ratio as signal
        df['Signal'] = df[f'{self.risk_metric.capitalize()}Ratio']
        
        # Warm-up
        df.iloc[:self.lookback_months, df.columns.get_loc('Signal')] = np.nan
        
        return df
