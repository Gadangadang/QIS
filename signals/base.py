"""Base classes for trading signal generation."""
from abc import ABC, abstractmethod
import pandas as pd


class SignalModel(ABC):
    """
    Abstract base class for all trading signal generators.
    
    All signal models must implement the generate() method which takes
    price data and returns a DataFrame with trading signals.
    
    Signal Convention:
        Position/Signal column values:
        - 1: Long position
        - 0: Flat (no position)
        - -1: Short position (if strategy supports shorting)
    
    Example:
        >>> class MySignal(SignalModel):
        ...     def generate(self, df):
        ...         df = df.copy()
        ...         df['Position'] = 1  # Always long
        ...         return df
    """
    
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from price data.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data (Open, High, Low, Close, Volume).
                             Must have 'Close' column at minimum.
        
        Returns:
            pd.DataFrame: Original DataFrame with added 'Position' or 'Signal' column.
                         Column should contain: 1 (long), 0 (flat), or -1 (short).
        
        Raises:
            NotImplementedError: If subclass doesn't implement this method.
        
        Note:
            - Return a copy of the input DataFrame to avoid side effects
            - Include any intermediate calculations as additional columns
            - Set warm-up/burn-in period positions to 0
        """
        raise NotImplementedError("Subclass must implement generate() method")

