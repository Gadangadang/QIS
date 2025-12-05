"""Mean reversion trading signals for counter-trend strategies."""
from typing import Optional
import pandas as pd
import numpy as np
from signals.base import SignalModel


class MeanReversionSignal(SignalModel):
    """
    Mean reversion signal generator using z-score of price deviations.
    
    Generates signals when price deviates significantly from its moving average:
    - Long when price < mean - (entry_z * std)
    - Short when price > mean + (entry_z * std)
    - Exit when price reverts to mean ± (exit_z * std)
    
    This strategy assumes prices will revert to their mean over time, making
    it suitable for range-bound assets like commodities or currencies.
    
    Attributes:
        window (int): Lookback period for mean and std calculation
        entry_z (float): Z-score threshold for entry (e.g., 2.0 = 2 standard deviations)
        exit_z (float): Z-score threshold for exit (e.g., 0.5 = half std dev from mean)
    
    Example:
        >>> signal = MeanReversionSignal(window=50, entry_z=2.0, exit_z=0.5)
        >>> df_with_signals = signal.generate(price_data)
        >>> # Long when price drops 2σ below mean, exit when within 0.5σ of mean
    """
    
    def __init__(self, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.5):
        """
        Initialize mean reversion signal generator.
        
        Args:
            window: Rolling window for mean and std calculation (default: 20)
            entry_z: Z-score threshold for entry signal (default: 2.0)
                    Entry triggers at ±entry_z standard deviations from mean
            exit_z: Z-score threshold for exit signal (default: 0.5)
                   Exits when price reverts to within ±exit_z std of mean
        
        Raises:
            ValueError: If window < 2, entry_z <= 0, exit_z < 0, or exit_z >= entry_z
        
        Note:
            Higher entry_z = fewer but stronger signals
            Lower exit_z = quicker exits (take profit sooner)
        """
        # Validate parameters (fail fast)
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if entry_z <= 0:
            raise ValueError(f"entry_z must be positive, got {entry_z}")
        if exit_z < 0:
            raise ValueError(f"exit_z must be non-negative, got {exit_z}")
        if exit_z >= entry_z:
            raise ValueError(f"exit_z ({exit_z}) must be < entry_z ({entry_z})")
        
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion signals using vectorized operations.
        
        Args:
            df: DataFrame with at least 'Close' column
        
        Returns:
            DataFrame with added columns:
                - Z: Z-score of price relative to rolling mean/std
                - Signal: Trading signal (1=long, -1=short, 0=flat)
        
        Raises:
            ValueError: If df is empty or missing 'Close' column
        
        Logic:
            Entry signals (when flat):
            - Z < -entry_z: Go long (oversold)
            - Z > +entry_z: Go short (overbought)
            
            Exit signals:
            - Long: Exit when Z > -exit_z (reverted) or Z > entry_z (stop loss)
            - Short: Exit when Z < exit_z (reverted) or Z < -entry_z (stop loss)
        
        Note:
            First 'window' bars will have Signal = 0 (insufficient data).
            Uses forward fill to maintain positions between signals.
        """
        # Validate input
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have 'Close' column")
        
        df = df.copy()
        close = df["Close"]
        
        # Calculate z-score (vectorized)
        sma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        df["Z"] = (close - sma) / std
        
        # Entry conditions (vectorized)
        long_entry = df["Z"] <= -self.entry_z  # Oversold
        short_entry = df["Z"] >= self.entry_z  # Overbought
        
        # Exit conditions (vectorized)
        long_exit_revert = df["Z"] >= -self.exit_z  # Mean reversion
        long_exit_stop = df["Z"] > self.entry_z  # Stop loss
        short_exit_revert = df["Z"] <= self.exit_z  # Mean reversion
        short_exit_stop = df["Z"] < -self.entry_z  # Stop loss
        
        # Initialize signal with entry signals
        df["Signal"] = 0
        df.loc[long_entry, "Signal"] = 1
        df.loc[short_entry, "Signal"] = -1
        
        # Forward fill to maintain positions
        df["Signal"] = df["Signal"].replace(0, np.nan).ffill().fillna(0).astype(int)
        
        # Apply exit signals (vectorized)
        # Exit long positions
        long_mask = df["Signal"] == 1
        exit_long = long_mask & (long_exit_revert | long_exit_stop)
        df.loc[exit_long, "Signal"] = 0
        
        # Exit short positions  
        short_mask = df["Signal"] == -1
        exit_short = short_mask & (short_exit_revert | short_exit_stop)
        df.loc[exit_short, "Signal"] = 0
        
        # Forward fill again after exits to prevent re-entry immediately
        df["Signal"] = df["Signal"].replace(0, np.nan).ffill().fillna(0).astype(int)
        
        # Clear warmup period
        df.iloc[:self.window, df.columns.get_loc("Signal")] = 0
        
        return df
