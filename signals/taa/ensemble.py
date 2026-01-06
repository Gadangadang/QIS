"""
TAA Ensemble Signal Generator.

Combines multiple signal types (momentum, carry, value, macro) into
unified expected return forecasts for portfolio optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from signals.base import SignalModel
from signals.taa.momentum import TimeSeriesMomentum, CrossSectionalMomentum
from signals.taa.carry import YieldCarry


class TAAEnsembleSignal(SignalModel):
    """
    Ensemble signal generator for TAA combining multiple signal sources.
    
    Combines momentum, carry, and other signals with configurable weights
    to produce unified expected return forecasts for optimization.
    
    Args:
        signals: Dict of signal generators {name: SignalModel}
        weights: Dict of signal weights {name: weight} (default: equal weights)
        combination_method: 'weighted_avg', 'rank_avg', or 'zscore_avg' (default: 'weighted_avg')
        normalize_signals: Normalize each signal before combining (default: True)
        min_signals: Minimum number of valid signals required (default: 1)
    
    Returns:
        DataFrame with 'Signal' column (ensemble expected return)
    
    Example:
        >>> from signals.taa import TimeSeriesMomentum, YieldCarry
        >>> ensemble = TAAEnsembleSignal(
        ...     signals={
        ...         'momentum': TimeSeriesMomentum(lookback_months=12),
        ...         'carry': YieldCarry(yield_type='spread')
        ...     },
        ...     weights={'momentum': 0.6, 'carry': 0.4}
        ... )
        >>> signals = ensemble.generate(monthly_data)
    """
    
    def __init__(
        self,
        signals: Dict[str, SignalModel],
        weights: Optional[Dict[str, float]] = None,
        combination_method: str = 'weighted_avg',
        normalize_signals: bool = True,
        min_signals: int = 1
    ):
        if not signals:
            raise ValueError("Must provide at least one signal generator")
        
        if combination_method not in ['weighted_avg', 'rank_avg', 'zscore_avg']:
            raise ValueError(
                f"combination_method must be 'weighted_avg', 'rank_avg', or 'zscore_avg', "
                f"got {combination_method}"
            )
        
        self.signals = signals
        self.combination_method = combination_method
        self.normalize_signals = normalize_signals
        self.min_signals = min_signals
        
        # Default to equal weights
        if weights is None:
            n = len(signals)
            self.weights = {name: 1.0/n for name in signals.keys()}
        else:
            # Validate weights
            if set(weights.keys()) != set(signals.keys()):
                raise ValueError("Weight keys must match signal keys")
            
            weight_sum = sum(weights.values())
            if not np.isclose(weight_sum, 1.0):
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
            
            self.weights = weights
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble signals.
        
        Args:
            df: Price DataFrame (passed to all signal generators)
        
        Returns:
            DataFrame with 'Signal' column and individual signal components
        """
        df = df.copy()
        
        # Generate all individual signals
        signal_results = {}
        for name, signal_gen in self.signals.items():
            result = signal_gen.generate(df)
            
            if 'Signal' not in result.columns:
                raise ValueError(f"Signal generator '{name}' did not return 'Signal' column")
            
            signal_results[name] = result['Signal']
        
        # Combine into DataFrame
        signals_df = pd.DataFrame(signal_results)
        
        # Normalize signals if requested
        if self.normalize_signals:
            if self.combination_method == 'zscore_avg':
                # Z-score normalization
                signals_norm = (signals_df - signals_df.mean()) / signals_df.std()
            elif self.combination_method == 'rank_avg':
                # Rank normalization (0 to 1)
                signals_norm = signals_df.rank(pct=True)
            else:  # weighted_avg
                # Simple standardization (mean 0, std 1)
                signals_norm = (signals_df - signals_df.mean()) / signals_df.std()
        else:
            signals_norm = signals_df
        
        # Apply weights and combine
        df['Signal'] = 0.0
        valid_signals = 0
        
        for name, weight in self.weights.items():
            if name in signals_norm.columns:
                # Add weighted signal (handle NaN)
                weighted_signal = signals_norm[name] * weight
                df['Signal'] = df['Signal'] + weighted_signal.fillna(0)
                
                # Count non-NaN signals
                valid_signals += (~signals_norm[name].isna()).astype(int)
                
                # Store individual signal for diagnostics
                df[f'Signal_{name}'] = signals_df[name]
        
        # Check minimum signals requirement
        if (valid_signals < self.min_signals).any():
            import warnings
            warnings.warn(
                f"Some rows have fewer than {self.min_signals} valid signals. "
                "Consider increasing lookback periods or filtering data."
            )
        
        return df


class MultiAssetEnsemble:
    """
    Ensemble signal generator for multiple assets simultaneously.
    
    Handles cross-sectional normalization and ranking across assets,
    which is critical for proper TAA signals.
    
    Args:
        signal_generators: Dict of signal generators to apply
        weights: Dict of signal weights (default: equal)
        cross_sectional_norm: Apply cross-sectional normalization (default: True)
    
    Example:
        >>> from signals.taa import TimeSeriesMomentum, YieldCarry
        >>> ensemble = MultiAssetEnsemble(
        ...     signal_generators={
        ...         'momentum': TimeSeriesMomentum(lookback_months=12),
        ...         'carry': YieldCarry(yield_type='spread')
        ...     },
        ...     weights={'momentum': 0.7, 'carry': 0.3}
        ... )
        >>> # Apply to dict of price DataFrames
        >>> signals_dict = ensemble.generate_multi_asset(prices_dict)
    """
    
    def __init__(
        self,
        signal_generators: Dict[str, SignalModel],
        weights: Optional[Dict[str, float]] = None,
        cross_sectional_norm: bool = True
    ):
        self.signal_generators = signal_generators
        self.cross_sectional_norm = cross_sectional_norm
        
        # Default to equal weights
        if weights is None:
            n = len(signal_generators)
            self.weights = {name: 1.0/n for name in signal_generators.keys()}
        else:
            self.weights = weights
    
    def generate_multi_asset(
        self, 
        prices: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate signals for multiple assets with cross-sectional normalization.
        
        Args:
            prices: Dict of {ticker: DataFrame} with price data
        
        Returns:
            Dict of {ticker: DataFrame} with 'Signal' column
        """
        # Step 1: Generate raw signals for each asset
        all_signals = {}
        
        for signal_name, signal_gen in self.signal_generators.items():
            signal_results = {}
            
            for ticker, df in prices.items():
                result = signal_gen.generate(df)
                signal_results[ticker] = result['Signal']
            
            # Convert to DataFrame (rows=dates, cols=tickers)
            all_signals[signal_name] = pd.DataFrame(signal_results)
        
        # Step 2: Cross-sectional normalization
        if self.cross_sectional_norm:
            normalized_signals = {}
            
            for signal_name, signal_df in all_signals.items():
                # Z-score across assets at each time point
                mean = signal_df.mean(axis=1, skipna=True)
                std = signal_df.std(axis=1, skipna=True)
                
                # Broadcast and normalize
                normalized = signal_df.sub(mean, axis=0).div(std, axis=0)
                normalized_signals[signal_name] = normalized
        else:
            normalized_signals = all_signals
        
        # Step 3: Combine signals with weights
        combined_signals = {}
        
        for ticker in prices.keys():
            ticker_signal = 0.0
            
            for signal_name, weight in self.weights.items():
                if signal_name in normalized_signals:
                    ticker_signal += normalized_signals[signal_name][ticker] * weight
            
            combined_signals[ticker] = ticker_signal
        
        # Step 4: Create output DataFrames
        output = {}
        
        for ticker, df in prices.items():
            result_df = df.copy()
            result_df['Signal'] = combined_signals[ticker]
            
            # Add individual signal components
            for signal_name in self.signal_generators.keys():
                if signal_name in all_signals:
                    result_df[f'Signal_{signal_name}'] = all_signals[signal_name][ticker]
            
            output[ticker] = result_df
        
        return output
