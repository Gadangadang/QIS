"""
Multi-Asset Signal Framework
Extends signal generation to work with multiple assets simultaneously.
"""
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
import numpy as np


class MultiAssetSignal(ABC):
    """
    Base class for signals that work across multiple assets.
    
    Takes dictionary of price DataFrames, returns dictionary of signal DataFrames.
    Each asset gets its own signal, enabling asset-specific strategy parameters.
    """
    
    @abstractmethod
    def generate(self, prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate signals for multiple assets.
        
        Args:
            prices: Dict mapping ticker -> DataFrame with OHLC data
            
        Returns:
            Dict mapping ticker -> DataFrame with Signal column (1=long, 0=flat, -1=short)
        """
        raise NotImplementedError


class SingleAssetWrapper(MultiAssetSignal):
    """
    Wraps a single-asset signal to work in multi-asset framework.
    
    Applies the same signal logic independently to each asset.
    """
    
    def __init__(self, signal_model):
        """
        Args:
            signal_model: Instance of SignalModel (e.g., MomentumSignalV2)
        """
        self.signal_model = signal_model
        
    def generate(self, prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply signal to each asset independently.
        
        Args:
            prices: Dict mapping ticker -> DataFrame with OHLC data
            
        Returns:
            Dict mapping ticker -> DataFrame with Signal column
        """
        signals = {}
        
        for ticker, df in prices.items():
            # Generate signal for this asset
            signal_df = self.signal_model.generate(df.copy())
            
            # Ensure 'Signal' column exists (convert from 'Position' if needed)
            if 'Signal' not in signal_df.columns:
                if 'Position' in signal_df.columns:
                    # Rename Position to Signal for consistency
                    signal_df = signal_df.rename(columns={'Position': 'Signal'})
                else:
                    raise ValueError(f"Signal model did not produce 'Signal' or 'Position' column for {ticker}")
            
            signals[ticker] = signal_df
        
        return signals


class UnifiedMomentumSignal(MultiAssetSignal):
    """
    Momentum signal applied uniformly across all assets.
    Uses same parameters for all assets (simpler, faster).
    """
    
    def __init__(self, lookback: int = 50, entry_z: float = 2.0, 
                 exit_z: float = 0.5, sma_period: int = 200):
        """
        Args:
            lookback: Lookback period for z-score calculation
            entry_z: Z-score threshold for entry
            exit_z: Z-score threshold for exit
            sma_period: SMA period for trend filter
        """
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.sma_period = sma_period
        
    def generate(self, prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate momentum signals for all assets."""
        signals = {}
        
        for ticker, df in prices.items():
            # Make a copy to avoid modifying original
            signal_df = df.copy()
            
            # Calculate returns
            signal_df['Return'] = signal_df['Close'].pct_change()
            
            # Calculate z-score
            rolling_mean = signal_df['Return'].rolling(self.lookback).mean()
            rolling_std = signal_df['Return'].rolling(self.lookback).std()
            signal_df['ZScore'] = (signal_df['Return'] - rolling_mean) / rolling_std
            
            # SMA trend filter
            signal_df['SMA'] = signal_df['Close'].rolling(self.sma_period).mean()
            signal_df['AboveSMA'] = (signal_df['Close'] > signal_df['SMA']).astype(int)
            
            # Generate raw signal
            signal_df['RawSignal'] = 0
            signal_df.loc[signal_df['ZScore'] > self.entry_z, 'RawSignal'] = 1   # Long on positive momentum
            signal_df.loc[signal_df['ZScore'] < -self.entry_z, 'RawSignal'] = -1  # Short on negative momentum
            
            # Exit when z-score crosses exit threshold
            signal_df.loc[signal_df['ZScore'].abs() < self.exit_z, 'RawSignal'] = 0
            
            # Forward-fill to hold positions
            signal_df['Signal'] = signal_df['RawSignal'].replace(0, np.nan).ffill().fillna(0)
            
            # Apply trend filter (only long when above SMA)
            signal_df.loc[signal_df['AboveSMA'] == 0, 'Signal'] = signal_df.loc[signal_df['AboveSMA'] == 0, 'Signal'].clip(upper=0)
            
            # Clean up
            signal_df = signal_df.drop(columns=['RawSignal'])
            
            signals[ticker] = signal_df
        
        return signals


class CrossAssetMomentumSignal(MultiAssetSignal):
    """
    Advanced: Momentum signal that considers cross-asset information.
    
    Example: If ES shows strong momentum, might increase NQ allocation.
    This is a placeholder for future development.
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        
    def generate(self, prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate cross-asset aware momentum signals.
        
        For now, just calculates individual momentum.
        Future: Add cross-correlation, regime detection, etc.
        """
        signals = {}
        
        # Step 1: Calculate momentum for each asset
        momentum_scores = {}
        for ticker, df in prices.items():
            returns = df['Close'].pct_change()
            momentum = returns.rolling(self.lookback).mean() / returns.rolling(self.lookback).std()
            momentum_scores[ticker] = momentum
        
        # Step 2: Generate signals (simple for now)
        for ticker, df in prices.items():
            signal_df = df.copy()
            
            # Get this asset's momentum
            momentum = momentum_scores[ticker]
            
            # Simple threshold-based signal
            signal_df['Signal'] = 0
            signal_df.loc[momentum > 1.0, 'Signal'] = 1
            signal_df.loc[momentum < -1.0, 'Signal'] = -1
            
            # Forward-fill positions
            signal_df['Signal'] = signal_df['Signal'].replace(0, np.nan).ffill().fillna(0)
            
            signals[ticker] = signal_df
        
        return signals


# Convenience function
def create_multi_asset_signal(signal_model) -> MultiAssetSignal:
    """
    Wrap a single-asset signal for multi-asset use.
    
    Args:
        signal_model: Instance of SignalModel (e.g., MomentumSignalV2)
        
    Returns:
        MultiAssetSignal instance
    """
    return SingleAssetWrapper(signal_model)


if __name__ == "__main__":
    # Test the framework
    from core.multi_asset_loader import load_assets
    from signals.momentum import MomentumSignalV2
    
    print("Testing MultiAssetSignal framework...")
    
    # Load data
    prices = load_assets(['ES', 'GC'], start_date='2020-01-01')
    
    # Test 1: SingleAssetWrapper with MomentumSignalV2
    print("\n" + "="*60)
    print("Test 1: SingleAssetWrapper")
    print("="*60)
    
    momentum_v2 = MomentumSignalV2(lookback=50, entry_z=2.0, exit_z=0.5, sma_period=200)
    wrapped_signal = SingleAssetWrapper(momentum_v2)
    signals = wrapped_signal.generate(prices)
    
    for ticker, df in signals.items():
        n_long = (df['Signal'] == 1).sum()
        n_short = (df['Signal'] == -1).sum()
        n_flat = (df['Signal'] == 0).sum()
        print(f"{ticker}: {n_long} long, {n_short} short, {n_flat} flat days")
    
    # Test 2: UnifiedMomentumSignal
    print("\n" + "="*60)
    print("Test 2: UnifiedMomentumSignal")
    print("="*60)
    
    unified_signal = UnifiedMomentumSignal(lookback=50, entry_z=2.0, exit_z=0.5, sma_period=200)
    signals = unified_signal.generate(prices)
    
    for ticker, df in signals.items():
        n_long = (df['Signal'] == 1).sum()
        n_short = (df['Signal'] == -1).sum()
        n_flat = (df['Signal'] == 0).sum()
        print(f"{ticker}: {n_long} long, {n_short} short, {n_flat} flat days")
