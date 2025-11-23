"""Mean reversion trading signals for counter-trend strategies."""
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
    
    def __init__(self, window=20, entry_z=2.0, exit_z=0.5):
        """
        Initialize mean reversion signal generator.
        
        Args:
            window (int): Rolling window for mean and std calculation. Default 20.
            entry_z (float): Z-score threshold for entry signal. Default 2.0.
                           Entry triggers at ±entry_z standard deviations from mean.
            exit_z (float): Z-score threshold for exit signal. Default 0.5.
                          Exits when price reverts to within ±exit_z std of mean.
        
        Note:
            Higher entry_z = fewer but stronger signals
            Lower exit_z = quicker exits (take profit sooner)
        """
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion signals for given price data.
        
        Args:
            df (pd.DataFrame): DataFrame with at least 'Close' column
        
        Returns:
            pd.DataFrame: Original DataFrame with added columns:
                - Z: Z-score of price relative to rolling mean/std
                - Position: Trading signal (1=long, -1=short, 0=flat)
        
        Logic:
            - If flat and Z < -entry_z: Go long (oversold)
            - If flat and Z > +entry_z: Go short (overbought)
            - If long and Z > -exit_z: Exit (price reverted to mean)
            - If short and Z < +exit_z: Exit (price reverted to mean)
            - Stop loss: Exit if price moves further against position
        
        Note:
            First 'window' bars will have Position = 0 (insufficient data).
        """
        df = df.copy()
        close = df["Close"]
        sma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        df["Z"] = (close - sma) / std

        # Initialize position column
        df["Position"] = 0
        
        # Build positions bar-by-bar
        for i in range(self.window, len(df)):
            prev_position = df.iloc[i - 1]["Position"]
            z_score = df.iloc[i]["Z"]
            
            # If flat, check for entry
            if prev_position == 0:
                if z_score <= -self.entry_z:
                    df.iloc[i, df.columns.get_loc("Position")] = 1  # Long entry
                elif z_score >= self.entry_z:
                    df.iloc[i, df.columns.get_loc("Position")] = -1  # Short entry
                else:
                    df.iloc[i, df.columns.get_loc("Position")] = 0  # Stay flat
            
            # If holding long, check for exit
            elif prev_position == 1:
                if z_score >= -self.exit_z:  # Mean reversion: exit when Z crosses back toward zero
                    df.iloc[i, df.columns.get_loc("Position")] = 0
                elif z_score > self.entry_z:  # Stop loss: price moved against us
                    df.iloc[i, df.columns.get_loc("Position")] = 0
                else:
                    df.iloc[i, df.columns.get_loc("Position")] = 1  # Hold
            
            # If holding short, check for exit
            elif prev_position == -1:
                if z_score <= self.exit_z:  # Mean reversion: exit when Z crosses back toward zero
                    df.iloc[i, df.columns.get_loc("Position")] = 0
                elif z_score < -self.entry_z:  # Stop loss: price moved against us
                    df.iloc[i, df.columns.get_loc("Position")] = 0
                else:
                    df.iloc[i, df.columns.get_loc("Position")] = -1  # Hold

        df["Position"] = df["Position"].astype(int)
        return df
