"""
Multi-Strategy Signal Framework
Allows different signal models for different assets in a portfolio.
"""
from typing import Dict
import pandas as pd
from core.multi_asset_signal import MultiAssetSignal


class MultiStrategySignal(MultiAssetSignal):
    """
    Apply different signal strategies to different assets.
    
    Example:
        strategies = {
            'ES': MomentumSignalV2(lookback=120, entry_threshold=0.02),
            'NQ': MomentumSignalV2(lookback=120, entry_threshold=0.02),
            'GC': MeanReversionSignal(lookback=50, entry_z=2.0)
        }
        multi_signal = MultiStrategySignal(strategies)
        signals = multi_signal.generate(prices)
    """
    
    def __init__(self, strategies: Dict[str, object]):
        """
        Initialize with asset-specific strategies.
        
        Args:
            strategies: Dict mapping ticker -> signal model instance
                       e.g., {'ES': momentum_model, 'GC': mean_reversion_model}
        """
        self.strategies = strategies
        
    def generate(self, prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate signals using asset-specific strategies.
        
        Args:
            prices: Dict mapping ticker -> DataFrame with OHLC data
            
        Returns:
            Dict mapping ticker -> DataFrame with Signal column
            
        Raises:
            ValueError: If asset in prices doesn't have a strategy defined
        """
        signals = {}
        
        for ticker, df in prices.items():
            if ticker not in self.strategies:
                raise ValueError(
                    f"No strategy defined for {ticker}. "
                    f"Available strategies: {list(self.strategies.keys())}"
                )
            
            # Get asset-specific strategy
            strategy = self.strategies[ticker]
            
            # Generate signal
            signal_df = strategy.generate(df)
            
            # Ensure Signal column exists
            if 'Position' in signal_df.columns and 'Signal' not in signal_df.columns:
                signal_df['Signal'] = signal_df['Position']
            elif 'Signal' not in signal_df.columns:
                raise ValueError(
                    f"Strategy for {ticker} did not produce 'Signal' or 'Position' column"
                )
            
            signals[ticker] = signal_df
        
        return signals
    
    def add_strategy(self, ticker: str, strategy: object):
        """
        Add or update strategy for a specific asset.
        
        Args:
            ticker: Asset ticker symbol
            strategy: Signal model instance
        """
        self.strategies[ticker] = strategy
    
    def remove_strategy(self, ticker: str):
        """Remove strategy for a specific asset."""
        if ticker in self.strategies:
            del self.strategies[ticker]
    
    def get_strategy(self, ticker: str):
        """Get strategy for a specific asset."""
        return self.strategies.get(ticker)
    
    def list_strategies(self) -> Dict[str, str]:
        """
        Get summary of strategies by asset.
        
        Returns:
            Dict mapping ticker -> strategy class name
        """
        return {
            ticker: type(strategy).__name__
            for ticker, strategy in self.strategies.items()
        }


class StrategyConfig:
    """
    Helper class to build multi-strategy configurations.
    
    Example:
        config = (StrategyConfig()
                  .add_momentum('ES', lookback=120)
                  .add_momentum('NQ', lookback=120)
                  .add_mean_reversion('GC', lookback=50))
        multi_signal = config.build()
    """
    
    def __init__(self):
        self.strategies = {}
    
    def add_momentum(self, ticker: str, lookback: int = 120, 
                     entry_threshold: float = 0.02, 
                     exit_threshold: float = -0.01,
                     sma_filter: int = 100):
        """
        Add momentum strategy for an asset.
        
        Args:
            ticker: Asset ticker
            lookback: Momentum lookback period
            entry_threshold: Return threshold for entry (e.g., 0.02 = 2%)
            exit_threshold: Return threshold for exit (e.g., -0.01 = -1%)
            sma_filter: SMA filter period
        """
        from signals.momentum import MomentumSignalV2
        
        self.strategies[ticker] = MomentumSignalV2(
            lookback=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            sma_filter=sma_filter
        )
        return self
    
    def add_mean_reversion(self, ticker: str, window: int = 20,
                          entry_z: float = 2.0, exit_z: float = 0.5):
        """
        Add mean reversion strategy for an asset.
        
        Args:
            ticker: Asset ticker
            window: Window for z-score calculation
            entry_z: Z-score threshold for entry (buy when < -entry_z, short when > entry_z)
            exit_z: Z-score threshold for exit
        """
        from signals.mean_reversion import MeanReversionSignal
        
        self.strategies[ticker] = MeanReversionSignal(
            window=window,
            entry_z=entry_z,
            exit_z=exit_z
        )
        return self
    
    def add_custom(self, ticker: str, strategy: object):
        """
        Add custom strategy instance for an asset.
        
        Args:
            ticker: Asset ticker
            strategy: Signal model instance
        """
        self.strategies[ticker] = strategy
        return self
    
    def build(self) -> MultiStrategySignal:
        """Build MultiStrategySignal from configuration."""
        if not self.strategies:
            raise ValueError("No strategies configured. Add at least one strategy.")
        return MultiStrategySignal(self.strategies)
    
    def summary(self) -> str:
        """Get readable summary of configuration."""
        lines = ["Strategy Configuration:"]
        for ticker, strategy in self.strategies.items():
            strategy_name = type(strategy).__name__
            lines.append(f"  {ticker}: {strategy_name}")
        return "\n".join(lines)
